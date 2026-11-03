; MimicCam installer (Inno Setup 6)
; Build the portable folder first:  python build.py
; Then compile:                     iscc installer.iss
;
; MimicCam and its source are Copyright (C) 2026 Ray Bekson (bigrayszn),
; https://github.com/bigrayszn/mimiccam - GPL-2.0-or-later.

#define AppName "MimicCam"
#define AppVersion "0.2.1"
#define AppPublisher "Ray Bekson (bigrayszn)"
#define AppUrl "https://github.com/bigrayszn/mimiccam"
#define AppExe "MimicCam.exe"

[Setup]
AppId={{8B0C6F3A-5D7E-4A16-9B0E-7C3F2A61D9C4}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename={#AppName}-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The virtual-camera filter has to register in HKLM, which needs elevation.
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "unitycam"; Description: "Install the bundled Unity Capture virtual camera (use this if OBS Studio is not installed)"; GroupDescription: "Virtual camera:"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "dist\MimicCam\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Optional: run tools\get_unitycapture.ps1 to vendor the MIT-licensed filter whose
; Install.bat carries the 32/64-bit DirectShow registration.
Source: "vendor\UnityCapture\*"; DestDir: "{tmp}\UnityCapture"; Tasks: unitycam; Flags: recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{tmp}\UnityCapture\Install.bat"; WorkingDir: "{tmp}\UnityCapture"; Tasks: unitycam; StatusMsg: "Registering the Unity Capture virtual camera..."; Flags: runhidden waituntilterminated
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
