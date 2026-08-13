import { createHash } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CACHE_TTL_MS,
  gcloudCredentialFingerprint,
  getGcloudToken,
  hasGcloudAdcCredentials,
  resetGcloudTokenCache,
} from "../src/gcloud-token.js";
import { useHermeticEnv } from "./test-helpers.js";

// HOME is included because getAdcPath falls back to it when GOOGLE_APPLICATION_CREDENTIALS
// is unset, which this suite exercises.
useHermeticEnv(["HOME"]);

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function writeAdcFile(body: unknown): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-gcloud-"));
  const path = join(dir, "adc.json");
  await writeFile(path, JSON.stringify(body), "utf8");
  return path;
}

afterEach(() => {
  resetGcloudTokenCache();
  vi.restoreAllMocks();
});

describe("getGcloudToken", () => {
  it("detects authorized_user ADC without exchanging a token", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi.spyOn(globalThis, "fetch");

    await expect(hasGcloudAdcCredentials()).resolves.toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.for([
    ["blank client_id", { client_id: "", client_secret: "secret", refresh_token: "refresh" }],
    ["blank client_secret", { client_id: "id", client_secret: "", refresh_token: "refresh" }],
    ["blank refresh_token", { client_id: "id", client_secret: "secret", refresh_token: "" }],
  ] as const)("rejects an authorized_user ADC file with a %s", async ([, fields]) => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({ type: "authorized_user", ...fields });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const fetchMock = vi.spyOn(globalThis, "fetch");

    await expect(hasGcloudAdcCredentials()).resolves.toBe(false);
    await expect(getGcloudToken()).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledWith(
      "LiteLLM gcloud auth: authorized_user ADC file is missing client_id, client_secret, or refresh_token.",
    );
  });

  it("exchanges authorized_user ADC credentials for an access token", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { access_token: "ya29.token" }));

    await expect(getGcloudToken()).resolves.toBe("ya29.token");

    expect(fetchMock).toHaveBeenCalledWith(
      "https://oauth2.googleapis.com/token",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: expect.stringContaining("grant_type=refresh_token"),
      }),
    );
  });

  it("uses the cached token until the TTL expires", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "first-token" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "second-token" }));

    const now = new Date("2026-06-10T12:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(now);
    await expect(getGcloudToken()).resolves.toBe("first-token");
    await expect(getGcloudToken()).resolves.toBe("first-token");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    vi.mocked(Date.now).mockReturnValue(now + CACHE_TTL_MS + 1);
    await expect(getGcloudToken()).resolves.toBe("second-token");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("reuses the cached token while the credential is unchanged", async () => {
    const adcPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "stable-client",
      client_secret: "stable-secret",
      refresh_token: "stable-refresh",
    });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { access_token: "cached" }));
    vi.spyOn(Date, "now").mockReturnValue(new Date("2026-06-10T12:00:00.000Z").getTime());
    process.env.GOOGLE_APPLICATION_CREDENTIALS = adcPath;

    await expect(getGcloudToken()).resolves.toBe("cached");
    await expect(getGcloudToken()).resolves.toBe("cached");
    await expect(getGcloudToken()).resolves.toBe("cached");

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  // Re-running `gcloud auth application-default login` rewrites refresh_token and leaves
  // client_id and client_secret alone, so the cache identity must react to that field on its
  // own. Rotating all three at once cannot show this.
  it("invalidates the cached token when only the refresh token rotates", async () => {
    const before = await writeAdcFile({
      type: "authorized_user",
      client_id: "same-client",
      client_secret: "same-secret",
      refresh_token: "refresh-one",
    });
    const after = await writeAdcFile({
      type: "authorized_user",
      client_id: "same-client",
      client_secret: "same-secret",
      refresh_token: "refresh-two",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "before-relogin" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "after-relogin" }));
    vi.spyOn(Date, "now").mockReturnValue(new Date("2026-06-10T12:00:00.000Z").getTime());

    process.env.GOOGLE_APPLICATION_CREDENTIALS = before;
    await expect(getGcloudToken()).resolves.toBe("before-relogin");

    process.env.GOOGLE_APPLICATION_CREDENTIALS = after;
    await expect(getGcloudToken()).resolves.toBe("after-relogin");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not reuse a cached token after the ADC identity changes", async () => {
    const firstPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "first-client",
      client_secret: "first-secret",
      refresh_token: "first-refresh",
    });
    const secondPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "second-client",
      client_secret: "second-secret",
      refresh_token: "second-refresh",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "first-token" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "second-token" }));
    vi.spyOn(Date, "now").mockReturnValue(new Date("2026-06-10T12:00:00.000Z").getTime());

    process.env.GOOGLE_APPLICATION_CREDENTIALS = firstPath;
    await expect(getGcloudToken()).resolves.toBe("first-token");

    process.env.GOOGLE_APPLICATION_CREDENTIALS = secondPath;
    await expect(getGcloudToken()).resolves.toBe("second-token");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns null and warns when no ADC file exists", async () => {
    delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
    delete process.env.APPDATA;
    process.env.HOME = await mkdtemp(join(tmpdir(), "pi-litellm-no-adc-"));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("No Google ADC file found"));
  });

  it("returns null and warns for service account credentials", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({ type: "service_account" });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Service account credentials are not supported"));
  });

  it("returns null when the token exchange fails", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("invalid_grant", { status: 400 }));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Token exchange failed"));
  });
});

describe("gcloudCredentialFingerprint", () => {
  const CLIENT = "fingerprint-client";
  const TOKEN = "fingerprint-refresh-token";

  it("is stable for the same credential within a process", () => {
    expect(gcloudCredentialFingerprint(CLIENT, TOKEN)).toBe(gcloudCredentialFingerprint(CLIENT, TOKEN));
  });

  it.for([
    ["the refresh token", CLIENT, "rotated-refresh-token"],
    ["the client id", "rotated-client", TOKEN],
  ] as const)("changes when %s changes", ([, clientId, refreshToken]) => {
    expect(gcloudCredentialFingerprint(clientId, refreshToken)).not.toBe(gcloudCredentialFingerprint(CLIENT, TOKEN));
  });

  it("retains no reversible credential material", () => {
    const fingerprint = gcloudCredentialFingerprint(CLIENT, TOKEN);

    expect(fingerprint).not.toContain(CLIENT);
    expect(fingerprint).not.toContain(TOKEN);
    expect(fingerprint).toMatch(/^[0-9a-f]{32}$/);
  });

  // A bare digest is non-reversible but linkable: anyone holding candidate tokens could
  // confirm a match, and the value would be identical in every process. Salting removes both.
  it("is not a bare digest of the credential", () => {
    const bare = createHash("sha256").update(`${CLIENT}\u0000${TOKEN}`).digest("hex").slice(0, 32);

    expect(gcloudCredentialFingerprint(CLIENT, TOKEN)).not.toBe(bare);
  });

  it("differs across module instances, so the salt is process-local", async () => {
    const first = gcloudCredentialFingerprint(CLIENT, TOKEN);
    vi.resetModules();
    const reloaded = await import("../src/gcloud-token.js");

    expect(reloaded.gcloudCredentialFingerprint(CLIENT, TOKEN)).not.toBe(first);
  });
});
