; SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
; SPDX-License-Identifier: GPL-2.0-or-later
; Throwaway harness: checks VirtualCameraNames against the strings the app
; really prints, without installing anything. Run it with /VERYSILENT /LOG and
; read the PASS/FAIL lines out of the log.
[Setup]
AppName=VcamParseTest
AppVersion=1.0
DefaultDirName={autopf}\VcamParseTest
PrivilegesRequired=lowest
Uninstallable=no
OutputDir=dist_test
OutputBaseFilename=vcam-parse-test
DisableProgramGroupPage=yes
Compression=none

[Code]
#include "vcam_check.issinc"

procedure LogCase(const Probe, Expected: String);
var
  Got: String;
begin
  Got := VirtualCameraNames(Probe);
  if Got = Expected then
    Log('PASS  probe="' + Probe + '"  ->  "' + Got + '"')
  else
    Log('FAIL  probe="' + Probe + '"  ->  "' + Got + '"  (expected "' + Expected + '")');
end;

// Runs the real probe against whatever MimicCam.exe sits in {app} - point
// /DIR at a stub that answers "none" to test the no-driver case.
procedure LogProbe(const Expected: String);
var
  Raw, Names: String;
begin
  Raw := ProbeVirtualCamera;
  Names := VirtualCameraNames(Raw);
  if Names = Expected then
    Log('PASS  probe exe raw="' + Trim(Raw) + '"  ->  "' + Names + '"')
  else
    Log('FAIL  probe exe raw="' + Trim(Raw) + '"  ->  "' + Names +
        '"  (expected "' + Expected + '")');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  LogProbe('');
  LogCase('Available virtual-camera backends: obs', 'obs');
  LogCase('Available virtual-camera backends: obs, unitycapture', 'obs, unitycapture');
  LogCase('Available virtual-camera backends: none', '');
  LogCase('No virtual camera backend found. Install OBS Studio (which provides ' +
          'the OBS Virtual Camera) or the bundled Unity Capture driver.', '');
  LogCase('Virtual-camera driver failed to load: [WinError 126] The specified ' +
          'module could not be found.', '');
  LogCase('', '');

  // With a stub that answers "none" in {app}, this is the no-driver branch: the
  // task dialog shows up and its buttons can be driven by a UI script.
  OfferVirtualCameraHelp;
end;
