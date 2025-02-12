# Dynaface Support

## Getting Version Information

When troubleshooting issues with Dynaface, your **macOS version**, **chip type**, and **memory size** are crucial details. To find this information:

1. Click the **Apple Menu** () in the top-left corner.
2. Select **About This Mac**.
3. A window will appear similar to the one below:

   ![Mac Version](https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/site/mac_info.jpg)

### Recommended System Requirements

- **Chip**: Ideally an Apple **M-series** (M1, M2, M3, M4). However, **Intel chips** are still supported.
- **Memory**: **16GB+** recommended (_8GB may work but is not ideal_).
- **macOS Version**: **macOS 14+** (Sonoma or later).

If you need support or are reporting a bug, please provide these three values.

---

## Dynaface Not Starting

If you double-click **Dynaface** and the application does not launch, you can use the **Terminal** to diagnose the issue.

### Launching Dynaface via Terminal

1. Click the **magnifying glass** (Spotlight) in the upper-right corner of your Mac screen.
2. Type **Terminal** and press **Enter** to open it.
3. Navigate to the folder where Dynaface is located. Common locations include:

   - **Downloads**: `cd ~/Downloads`
   - **Desktop**: `cd ~/Desktop`

   Issue the following command to move to the correct directory.

   ```sh
   cd ~/Downloads
   ```

   You can list the files in the directory with:

   ```sh
   ls
   ```

4. Dynaface will be listed as `Dynaface.app`. Start it manually with:

   ```sh
   ./Dynaface.app/Contents/MacOS/dynaface
   ```

This will generate output in the terminal, which may include an error message. If you need support, **copy and paste** this output when submitting a bug report.

Example terminal output:

![Mac Terminal](https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/images/facial/site/mac_terminal.jpg)

---

## Dynaface Log Files

If Dynaface launches but you experience issues, the **log files** may contain useful information.

### Accessing Log Files

1. Open Dynaface.
2. Click **Help** → **Open Support Logs**.

Alternatively, you can manually access logs:

1. Click **Go** in the Finder menu.
2. Select **Go to Folder**.
3. Enter the following path:

   ```sh
   ~/Library/Containers/com.heatonresearch.dynaface/Data/logs
   ```

Including log files with your bug report will help diagnose the issue more effectively.

---

### Need More Help?

If you're still having trouble, please provide:

- macOS version, chip type, and memory size.
- Terminal output (if applicable).
- Dynaface log files.

This will help us resolve your issue faster. 🚀
