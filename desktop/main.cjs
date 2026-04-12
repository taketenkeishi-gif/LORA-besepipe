const { app, BrowserWindow } = require("electron");
const http = require("http");

const FRONTEND_URL = "http://127.0.0.1:5173";

function waitForFrontend(maxRetries = 60, intervalMs = 500) {
  return new Promise((resolve, reject) => {
    let retries = 0;

    const check = () => {
      const req = http.get(FRONTEND_URL, (res) => {
        res.resume();
        if (res.statusCode >= 200 && res.statusCode < 500) {
          resolve();
          return;
        }
        retry();
      });
      req.on("error", retry);
      req.setTimeout(1000, () => {
        req.destroy();
        retry();
      });
    };

    const retry = () => {
      retries += 1;
      if (retries >= maxRetries) {
        reject(new Error("Frontend did not start in time"));
        return;
      }
      setTimeout(check, intervalMs);
    };

    check();
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  win.webContents.session.clearCache().catch(() => {});
  win.loadURL(FRONTEND_URL);
}

app.whenReady().then(async () => {
  try {
    await waitForFrontend();
    createWindow();
  } catch (err) {
    console.error("[Desktop] Failed to connect frontend:", err.message);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  app.quit();
});
