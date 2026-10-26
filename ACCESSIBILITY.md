# Accessibility

Dynaface should be usable by everyone who needs it, including people who use a screen reader, rely on a keyboard instead of a mouse, or have low vision or color blindness. This page describes where things stand and how to report a problem.

## Where things stand

- **The libraries** (Python and .NET) have no user interface. Their results are plain numbers that you can use in any tool.
- **The desktop app** is built with Qt, which gives its standard controls, such as menus, buttons, tabs and dialogs, basic support for keyboard navigation and the operating system's screen reader. The menus have the usual keyboard shortcuts for actions such as Open, Save, Print, Undo, Copy and Paste.
- **Measurements can be exported** from the app to CSV or Excel, so the data is available without the charts and image overlays.

## Known limitations

- Face images, landmark overlays and charts are visual only, with no text descriptions.
- The app doesn't set its own screen reader labels yet, so controls without visible text may not be announced clearly.
- The app hasn't had a formal accessibility review, including for color contrast.

## Reporting a problem

If part of Dynaface is hard to use with your assistive technology or setup, please open an issue with the [Bug report form](https://github.com/jeffheaton/dynaface/issues/new/choose). Say that it's an accessibility problem, and include your platform and any assistive technology you use. If you'd rather not post publicly, email jeff@jeffheaton.com.

Accessibility problems are treated as bugs, and pull requests that fix them are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
