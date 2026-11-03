; SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
; SPDX-License-Identifier: GPL-2.0-or-later
; MimicCam 0.2.4 - single-file setup
; MimicCam and its source are Copyright (C) 2026 Ray Bekson (bigrayszn),
; https://github.com/bigrayszn/mimiccam - GPL-2.0-or-later.
; Builds one .exe that carries the portable build, all five ONNX models, the
; character photos, the studio plates and settings. Double-click on the target
; PC, no PowerShell and no admin rights.
;
; Compile with:  iscc setup.iss
; Optional:      iscc /O"out dir" setup.iss

#define AppName        "MimicCam"
#define AppVersion     "0.2.4"
#define AppPublisher   "Ray Bekson (bigrayszn)"
#define AppUrl         "https://github.com/bigrayszn/mimiccam"
#define AppExe         "MimicCam.exe"

; The verified PyInstaller output (short path on purpose: its DLLs need one).
#define AppSource      "C:\MimicCam-dist\MimicCam"
; The export folder built earlier in this workspace.
#define DataSource     "C:\Users\offic\.config\freebuff-desktop\local-projects\2ba4c2c9-6ffa-4807-9ee5-adbff78c99c5\threads\caeb8bc427fc350c860c96235c13612bb1302ef149dfac46ec6b405eec76f2a3\exports\MimicCam-export\userdata"

[Setup]
AppId={{9F1D6A24-7C3B-4E58-9B2A-5D0E8C41B7F3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion=0.2.4.0
VersionInfoDescription={#AppName} setup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile=..\assets\mimiccam.ico
; Per-user install: no UAC prompt, nothing written to Program Files. On the
; target PC this lands in %LOCALAPPDATA%\Programs\MimicCam, which is short
; enough for Windows to load the native camera module.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
Compression=lzma2/normal
SolidCompression=no
OutputDir=dist
OutputBaseFilename={#AppName}-{#AppVersion}-setup
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
; The app itself.
Source: "{#AppSource}\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#AppSource}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; Offline models (~908 MB): without these the app would download them again.
Source: "{#DataSource}\models\*"; DestDir: "{code:GetDataDir}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
; Character photos and the cached room plates built from them.
Source: "{#DataSource}\characters\*"; DestDir: "{code:GetDataDir}\characters"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#DataSource}\studio\*"; DestDir: "{code:GetDataDir}\studio"; Flags: ignoreversion recursesubdirs createallsubdirs
; Settings keep the prompt, preset and camera choice. onlyifdoesntexist so a
; reinstall never overwrites what the user has configured since.
Source: "{#DataSource}\settings.json"; DestDir: "{code:GetDataDir}"; Flags: ignoreversion onlyifdoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The registry-free Decart key lives in Credential Manager, not here, so this
; is only the local model cache and settings.
Type: filesandordirs; Name: "{code:GetDataDir}\models"

[Code]
const
  // Windows refuses to load a DLL when the full path nears MAX_PATH. This is
  // the longest name inside the bundle, so it decides the budget.
  NativeLeaf = '\_internal\pyvirtualcam\_native_windows_unity_capture.cp310-win_amd64.pyd';
  PathBudget = 228;

var
  DataDirOverride: String;
  // True when this PC already had settings.json before the install: its
  // camera_index is then the user's own choice and must not be overridden.
  SettingsExisted: Boolean;

function InitializeSetup(): Boolean;
var
  I: Integer;
  P: String;
begin
  { /DATA="D:\somewhere" redirects the user-data target. Used by the automated
    install test so it never touches a real %APPDATA%\MimicCam. }
  DataDirOverride := '';
  for I := 1 to ParamCount do
  begin
    P := ParamStr(I);
    if CompareText(Copy(P, 1, 6), '/DATA=') = 0 then
      DataDirOverride := Copy(P, 7, Length(P) - 6);
  end;
  // Checked before anything is copied: afterwards the file always exists.
  // The path is built inline rather than through GetDataDir, which is declared
  // further down.
  if DataDirOverride <> '' then
    SettingsExisted := FileExists(DataDirOverride + '\settings.json')
  else
    SettingsExisted := FileExists(ExpandConstant('{userappdata}\MimicCam\settings.json'));
  Result := True;
end;

function GetDataDir(Param: String): String;
begin
  if DataDirOverride <> '' then
    Result := DataDirOverride
  else
    Result := ExpandConstant('{userappdata}\MimicCam');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Deepest: Integer;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Deepest := Length(WizardDirValue) + Length(NativeLeaf);
    if Deepest > PathBudget then
    begin
      MsgBox('That folder is too deep for Windows to load MimicCam''s virtual' + #13#10 +
             'camera module.' + #13#10#13#10 +
             'The deepest installed file would be ' + IntToStr(Deepest) +
             ' characters long; Windows stops loading DLLs near 240.' + #13#10#13#10 +
             'Please choose a shorter folder, for example C:\MimicCam' + #13#10 +
             '(that one needs administrator rights - the installer runs without' + #13#10 +
             'them, so %LOCALAPPDATA%\Programs\MimicCam is the default).',
             mbError, MB_OK);
      Result := False;
    end;
  end;
end;

{ The stored character-photo path is absolute and names the machine the export
  was made on. Point it at the copy this install just placed in the user's
  profile. Written as UTF-8 without a BOM - the app parses settings.json as
  plain UTF-8 and silently falls back to defaults if it finds a BOM. }
procedure FixReferenceImage;
var
  Target, Text, Key, OldVal, BaseName, NewDir, NewVal: String;
  Lines: TArrayOfString;
  KeyPos, ValStart, QuotePos, I, LastSlash: Integer;
begin
  Target := GetDataDir('') + '\settings.json';
  if not FileExists(Target) then
    Exit;
  if not LoadStringsFromFile(Target, Lines) then
    Exit;

  { The file is pretty-printed JSON. Work on it as one string: JSON does not
    care about the line breaks, and one line is simpler to patch. }
  Text := '';
  for I := 0 to GetArrayLength(Lines) - 1 do
  begin
    if I > 0 then
      Text := Text + #13#10;
    Text := Text + Lines[I];
  end;

  Key := '"reference_image": "';
  KeyPos := Pos(Key, Text);
  if KeyPos = 0 then
    Exit;
  ValStart := KeyPos + Length(Key);

  // The value is a JSON string; find its closing quote. Nothing inside a
  // Windows path is escaped, so the first quote after the start is the end.
  QuotePos := 0;
  I := ValStart;
  while I <= Length(Text) do
  begin
    if Text[I] = '"' then
    begin
      QuotePos := I;
      Break;
    end;
    I := I + 1;
  end;
  if QuotePos = 0 then
    Exit;

  OldVal := Copy(Text, ValStart, QuotePos - ValStart);
  if OldVal = '' then
    Exit;

  // Basename: everything after the last backslash. JSON stores each backslash
  // doubled, which makes no difference to the search.
  LastSlash := 0;
  for I := 1 to Length(OldVal) do
    if OldVal[I] = '\' then
      LastSlash := I;
  if LastSlash = 0 then
    Exit;
  BaseName := Copy(OldVal, LastSlash + 1, Length(OldVal) - LastSlash);

  NewDir := GetDataDir('') + '\characters\';
  StringChangeEx(NewDir, '\', '\\', True);
  NewVal := NewDir + BaseName;

  if OldVal = NewVal then
    Exit;
  StringChangeEx(Text, OldVal, NewVal, True);

  SetArrayLength(Lines, 1);
  Lines[0] := Text;
  SaveStringsToUTF8FileWithoutBOM(Target, Lines, False);
end;

#include "camera_pick.issinc"
#include "vcam_check.issinc"

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    FixReferenceImage;
    AutoPickCamera;
    OfferVirtualCameraHelp;
  end;
end;
