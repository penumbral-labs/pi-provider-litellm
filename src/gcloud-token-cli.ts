// Pi loads the extension through its TypeScript loader, but this helper runs
// under plain Node type stripping; keep this file and its target graph erasable.
const moduleUrl = process.argv[2];
if (!moduleUrl) {
  throw new Error("Google ADC token module URL is required");
}

const { getGcloudToken } = (await import(moduleUrl)) as {
  getGcloudToken(): Promise<string | null>;
};
const token = await getGcloudToken();
if (token) {
  process.stdout.write(token);
} else {
  process.exitCode = 1;
}
