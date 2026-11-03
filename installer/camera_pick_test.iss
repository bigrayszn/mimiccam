; SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
; SPDX-License-Identifier: GPL-2.0-or-later
; Throwaway harness: exercises the installer's camera auto-pick without
; installing anything. Compile and run with /VERYSILENT /LOG, then read the
; PASS/FAIL lines out of the log.
;
; Two halves:
;   1. parser cases, synthetic strings shaped like the app's real output;
;   2. the real thing - runs MimicCam.exe --list-cameras from {app} and rewrites
;      a copy of the shipped settings.json with the chosen index.
[Setup]
AppName=CamPickTest
AppVersion=1.0
DefaultDirName={autopf}\CamPickTest
PrivilegesRequired=lowest
Uninstallable=no
OutputDir=dist_test
OutputBaseFilename=cam-pick-test
DisableProgramGroupPage=yes
Compression=none

[Files]
Source: "..\exports\MimicCam-export\userdata\settings.json"; DestDir: "{tmp}"; DestName: "settings.json"

[Code]
// The include's AutoPickCamera asks setup.iss where the data lives; here it is
// only compiled, never run, so a fixed path is enough.
function GetDataDir(Param: String): String;
begin
  Result := ExpandConstant('{tmp}\data');
end;

// Set by setup.iss's InitializeSetup from the real data folder.
var
  SettingsExisted: Boolean;

#include "camera_pick.issinc"

procedure LogPick(const Name, Probe: String);
var
  Idx, Width, Height, Luma: Integer;
  Got: String;
begin
  if BestCameraFromProbe(Probe, Idx, Width, Height, Luma) then
    Got := IntToStr(Idx) + ' ' + IntToStr(Width) + 'x' + IntToStr(Height) +
           ' luma ' + IntToStr(Luma)
  else
    Got := 'none';
  Log('CASE ' + Name + ' -> ' + Got);
end;

procedure LogCase(const Name, Probe, Expected: String);
var
  Idx, Width, Height, Luma: Integer;
  Got: String;
begin
  if BestCameraFromProbe(Probe, Idx, Width, Height, Luma) then
    Got := IntToStr(Idx) + ' ' + IntToStr(Width) + 'x' + IntToStr(Height) +
           ' luma ' + IntToStr(Luma)
  else
    Got := 'none';
  if Got = Expected then
    Log('PASS  ' + Name + ' -> ' + Got)
  else
    Log('FAIL  ' + Name + ' -> ' + Got + '  (expected ' + Expected + ')');
end;

// The whole file as one string, the way the installer works on it.
function ReadAll(const Path: String): String;
var
  Lines: TArrayOfString;
  I: Integer;
begin
  Result := '';
  if not LoadStringsFromFile(Path, Lines) then
    Exit;
  for I := 0 to GetArrayLength(Lines) - 1 do
  begin
    if I > 0 then
      Result := Result + #13#10;
    Result := Result + Lines[I];
  end;
end;

// The real settings file shipped in the installer, plus whatever index we
// decide on: checks that only the number changes, that the prompt survives and
// that no BOM is written.
procedure CheckSettingsRewrite(const Path: String; NewIndex: Integer);
var
  Before, After: String;
  Ok: Boolean;
begin
  if not FileExists(Path) then
  begin
    Log('FAIL  settings fixture missing: ' + Path);
    Exit;
  end;
  Before := ReadAll(Path);

  Ok := SetJsonInt(Path, 'camera_index', NewIndex);
  After := ReadAll(Path);

  if not Ok then
    Log('FAIL  SetJsonInt returned False')
  else if Pos('"camera_index": ' + IntToStr(NewIndex), After) = 0 then
    Log('FAIL  camera_index did not become ' + IntToStr(NewIndex))
  else if Length(After) <> Length(Before) + Length(IntToStr(NewIndex)) - 1 then
    Log('FAIL  file length changed by the wrong amount (' + IntToStr(Length(Before)) +
        ' -> ' + IntToStr(Length(After)) + ')')
  else if After[1] <> '{' then
    Log('FAIL  file does not start with "{"; first char code ' +
        IntToStr(Ord(After[1])) + ' (BOM?)')
  else if Pos('"prompt":', After) = 0 then
    Log('FAIL  the prompt is gone from settings.json')
  else if Pos('"reference_image":', After) = 0 then
    Log('FAIL  reference_image is gone from settings.json')
  else
    Log('PASS  settings.json rewritten: camera_index ' + IntToStr(NewIndex) +
        ', ' + IntToStr(Length(After)) + ' chars, no BOM, other keys intact');

  // Idempotent: writing the same number again must not touch the file.
  if SetJsonInt(Path, 'camera_index', NewIndex) then
    Log('PASS  second write of the same value is a no-op')
  else
    Log('FAIL  second write of the same value failed');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Raw, RawFlat: String;
  Idx, Width, Height, Luma: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;

  // -- parser cases -------------------------------------------------------
  LogCase('real webcam 1080p beside a dark 720p device',
          'index=0 1280x720 luma=0.0'#10'index=1 1920x1080 luma=51.5',
          '1 1920x1080 luma 51');
  LogCase('dark 4K next to a lit 720p',
          'index=3 3840x2160 luma=0.2'#10'index=0 1280x720 luma=40.0',
          '0 1280x720 luma 40');
  LogCase('two lit: bigger wins',
          'index=0 640x480 luma=50.0'#10'index=2 1920x1080 luma=30.0',
          '2 1920x1080 luma 30');
  LogCase('two identical lit: lower index wins',
          'index=4 1280x720 luma=20.0'#10'index=1 1280x720 luma=20.0',
          '1 1280x720 luma 20');
  LogCase('only a dark camera: still better than nothing',
          'index=2 640x480 luma=0.4', '2 640x480 luma 0');
  LogCase('the no-camera message is not a camera',
          'No working camera found.', 'none');
  LogCase('empty output', '', 'none');
  LogCase('warning line mixed in with a camera',
          'warning: DSHOW could not load'#10'index=1 1280x720 luma=12.0',
          '1 1280x720 luma 12');
  LogCase('comma decimal separator',
          'index=1 1280x720 luma=51,5', '1 1280x720 luma 51');

  // -- the real probe on this PC -----------------------------------------
  Raw := ProbeCameras;
  RawFlat := Trim(Raw);
  StringChangeEx(RawFlat, #10, ' | ', True);
  Log('RAW PROBE: ' + RawFlat);
  if BestCameraFromProbe(Raw, Idx, Width, Height, Luma) then
    Log('CASE live probe -> ' + IntToStr(Idx) + ' ' + IntToStr(Width) + 'x' +
        IntToStr(Height) + ' luma ' + IntToStr(Luma))
  else
    Log('CASE live probe -> none');

  // -- settings rewrite ---------------------------------------------------
  CheckSettingsRewrite(ExpandConstant('{tmp}\settings.json'), 1);
  CheckSettingsRewrite(ExpandConstant('{tmp}\settings.json'), 3);
  CheckSettingsRewrite(ExpandConstant('{tmp}\settings.json'), 0);
end;
