; SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
; SPDX-License-Identifier: GPL-2.0-or-later
; MimicCam 0.2.3 - installer for public release.
; MimicCam and its source are Copyright (C) 2026 Ray Bekson (bigrayszn),
; https://github.com/bigrayszn/mimiccam - GPL-2.0-or-later.
;
; Unlike installer\setup.iss this one carries the application only: no ONNX
; models, no character photos, no settings. The models (~908 MB) download once
; from inside the app, which is also how a new user gets a legal chance to read
; the InsightFace research licence before accepting it.
;
; Build the portable folder first:
;   python build.py --dist C:\MimicCam-dist4
; then compile:
;   iscc /O"C:\MimicCam-dist4\release" installer\public.iss

#define AppName        "MimicCam"
#define AppVersion     "0.2.3"
#define AppPublisher   "Ray Bekson (bigrayszn)"
#define AppUrl         "https://github.com/bigrayszn/mimiccam"
#define AppExe         "MimicCam.exe"
#define AppSource      "C:\MimicCam-dist4\MimicCam"
#define RepoRoot       "C:\Users\offic\.config\freebuff-desktop\local-projects\2ba4c2c9-6ffa-4807-9ee5-adbff78c99c5\threads\caeb8bc427fc350c860c96235c13612bb1302ef149dfac46ec6b405eec76f2a3"

[Setup]
AppId={{9F1D6A24-7C3B-4E58-9B2A-5D0E8C41B7F3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion=0.2.3.0
VersionInfoDescription={#AppName} setup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile={#RepoRoot}\assets\mimiccam.ico
; Per-user install: no UAC prompt. The default lands in
; %LOCALAPPDATA%\Programs\MimicCam, which is short enough for Windows to load
; the native virtual-camera module (see the path budget in build.py).
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
Compression=lzma2/normal
SolidCompression=no
OutputDir=C:\MimicCam-dist4\release
OutputBaseFilename={#AppName}-{#AppVersion}-setup
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#AppSource}\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#AppSource}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs ignoreversion
Source: "{#RepoRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\INSTALL.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
; The ownership documents travel with the install, so a copy found on a machine
; still says whose it is even when the download is long gone.
Source: "{#RepoRoot}\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\COPYRIGHT.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\TRADEMARK.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autoprograms}\Install guide"; Filename: "{app}\INSTALL.md"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Models and settings live in %APPDATA%\MimicCam so an update keeps them; say so
; in the uninstaller rather than deleting a 908 MB download behind the user.
Type: filesandordirs; Name: "{app}"

[Code]
// The probe helpers live in vcam_check.issinc and have to be included inside
// the [Code] section, exactly as setup.iss does. Note: comments in [Code] use
// // or {}, never the ; that the other sections use.
#include "vcam_check.issinc"

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    OfferVirtualCameraHelp;
end;
