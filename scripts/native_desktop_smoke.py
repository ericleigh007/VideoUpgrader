from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from appium import webdriver
from appium.options.windows import WindowsOptions
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common.exceptions import NoSuchElementException, WebDriverException


def build_options(app_path: str | None, top_level_window: str | None) -> WindowsOptions:
    options = WindowsOptions()
    options.platform_name = "Windows"
    options.automation_name = "NovaWindows"
    options.set_capability("appium:newCommandTimeout", 120)
    options.set_capability("appium:createSessionTimeout", 30000)
    if top_level_window:
      options.set_capability("appium:appTopLevelWindow", top_level_window)
    elif app_path:
      options.app = app_path
    else:
      raise ValueError("Either app_path or top_level_window must be provided")
    return options


def find_named_element(driver: webdriver.Remote, name: str) -> dict[str, object] | None:
    try:
        element = driver.find_element(AppiumBy.NAME, name)
    except NoSuchElementException:
        return None

    return {
        "name": name,
        "rect": element.rect,
        "enabled": element.is_enabled(),
        "displayed": element.is_displayed(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:4723")
    parser.add_argument("--app-path")
    parser.add_argument("--top-level-window")
    parser.add_argument("--output", default="artifacts/runtime/native-desktop-smoke.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver: webdriver.Remote | None = None
    started_at = time.time()
    diagnostics: dict[str, object] = {
        "startedAt": started_at,
        "server": args.server,
        "appPath": args.app_path,
        "topLevelWindow": args.top_level_window,
    }

    try:
        options = build_options(args.app_path, args.top_level_window)
        driver = webdriver.Remote(args.server, options=options)
        driver.implicitly_wait(2)
        time.sleep(2)

        diagnostics["title"] = driver.title
        screenshot_path = output_path.with_suffix(".png")
        driver.get_screenshot_as_file(str(screenshot_path))
        diagnostics["screenshotPath"] = str(screenshot_path)
        diagnostics["controls"] = {
            name: find_named_element(driver, name)
            for name in ["Select Video", "Run Upscale", "Open Source Externally", "Play", "Pause", "Restart"]
        }

        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(json.dumps(diagnostics, indent=2))
        return 0
    except WebDriverException as error:
        diagnostics["error"] = str(error)
        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(json.dumps(diagnostics, indent=2), file=sys.stderr)
        return 1
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())