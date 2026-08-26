## Download

| | File | Requirements |
|---|---|---|
| **macOS** | `SoulFont-…-macos-arm64.dmg` | Apple Silicon (M1 or newer) |
| **Windows** | `SoulFont-…-windows-x64.zip` | Windows 10 or 11, 64-bit |

Nothing else to install — no Python, no Poppler. The model ships inside the app.

Intel Macs are not supported.

## Installing

**macOS.** Open the `.dmg` and drag Soul Font to Applications. The first launch is
blocked because the app is not signed: macOS says it cannot be opened. Go to
**System Settings → Privacy & Security**, scroll down, and press **Open Anyway**. It only
asks once.

**Windows.** Extract the `.zip` anywhere and run `SoulFont.exe`. SmartScreen shows
"Windows protected your PC" for the same reason — press **More info**, then **Run anyway**.

Both warnings mean the build is unsigned, not that anything is wrong with it. Verify the
download against `SHA256SUMS.txt` if you want to be sure it arrived intact:

```
shasum -a 256 -c SHA256SUMS.txt      # macOS
certutil -hashfile <file> SHA256      # Windows
```

## Using it

1. Sign up. The account is local to your machine — nothing is uploaded, ever.
2. **New Font** → download the template PDF.
3. **Print it, write all 130 cells by hand, and scan it back to a single PDF.**
4. Upload the scan and name your font.
5. Wait. Every one of the 11,172 Korean syllables is drawn, so this takes roughly 15-20
   minutes on a machine with a GPU and 20-40 without. The progress bar shows the count as
   it goes.
6. Download the Light / Regular / Bold family.

Closing Soul Font stops a build in progress, and it starts over next time.

Your fonts and templates live in `~/Library/Application Support/Soul Font` on macOS and
`%LOCALAPPDATA%\Soul Font` on Windows. Uninstalling is deleting the app and that folder.
