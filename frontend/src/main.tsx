import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "@radix-ui/themes/styles.css";
import { Theme } from "@radix-ui/themes";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Theme appearance="dark" accentColor="iris" grayColor="slate" radius="medium" scaling="95%"><App /></Theme>
  </React.StrictMode>
);

