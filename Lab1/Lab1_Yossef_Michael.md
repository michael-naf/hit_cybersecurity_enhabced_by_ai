## **_CTI Report Mapping_**

### **_Students:_**

·       Yossef Okropiridze, <yossiu1@gmail.com>

·       Michael Naftalishen, <michaelnafi99@gmail.com>

### **_Link to the Article:_**

[Exploitation of CLFS zero-day leads to ransomware activity \| Microsoft
Security
Blog](https://www.microsoft.com/en-us/security/blog/2025/04/08/exploitation-of-clfs-zero-day-leads-to-ransomware-activity/)

### **_Schema Description:_**

An adversary group exploited a zero‑day kernel vulnerability in the
Windows Common Log File System (CLFS) to escalate from a standard user
to system privileges. It hasn\'t been determined what initial access
vectors led to the devices being compromised by adversary group but
afterwards, [they delivered a malicious MSBuild payload downloaded via
*[certutil]{.underline}*]{.mark}, [used the CLFS exploit]{.mark} to
[overwrite a process token]{.mark} and [inject into high‑privilege
processes]{.mark}, [dumped LSASS memory with *[procdump]{.underline}* to
harvest credentials]{.mark}, [and then deployed ransomware that deleted
backups]{.mark}, [cleared event logs]{.mark} and [encrypted files
(ransom note !\_READ_ME_REXX2\_!.txt)]{.mark}.

![Sequence of events](Lab1_Schema.jpg)

### **_Tactics, Techniques and Behaviors:_**

- **Command and Control: T1105 --- Ingress Tool Transfer:**  
  Observation: The adversaries used certutil to download an MSBuild file
  (hosted on a compromised third‑party site) that contained an encrypted
  payload.

- **Defense Evasion: T1140 --- Deobfuscate/Decode Files or
  Information:**  
  Observation: The downloaded MSBuild file contained an encrypted
  payload that was decoded/processed at runtime (MSBuild + certutil used
  to deliver and activate the payload).

<!-- -->

- **Defense Evasion: T1211 --- Exploitation for Defense
  Evasion:**  
  Observation: The exploit targets a vulnerability in the CLFS kernel
  driver. It's notable that the exploit first uses the
  *NtQuerySystemInformation* API to leak kernel addresses to user mode.

- **Defense Evasion, Privilege Escalation: T1134 --- Access
  Token Manipulation:**  
  Observation: The exploit overwrote the process token (article notes
  token bits set to 0xFFFFFFFF), effectively granting full privileges to
  the process - matching access‑token manipulation/impersonation
  behaviors.

<!-- -->

- **Defense Evasion, Privilege Escalation: T1055 --- Process
  Injection (and sub-techniques):**  
  Observation: After exploitation, the attackers injected into
  winlogon.exe (process injection / in‑memory techniques) to execute in
  a high‑privilege context.

<!-- -->

- **Credential Access: T1003.001 --- OS Credential Dumping:
  LSASS Memory:**  
  Observation: The threat actors used procdump.exe to dump LSASS memory
  to harvest credentials (Sysinternals procdump -ma lsass.exe).

<!-- -->

- **Impact: T1490 --- Inhibit System Recovery:**  
  Observation: Commands recorded include deleting backup catalogs
  (*[wbadmin]{.underline}* delete catalog -quiet) and disabling recovery
  options - typical ransomware behavior to prevent restoration.

- **Defense Evasion: T1070.001 --- Indicator Removal: Clear
  Windows Event Logs:**  
  Observation: The actors ran *[wevtutil]{.underline}* cl Application to
  clear event logs and remove forensic evidence.

- **Impact: T1486 --- Data Encrypted for Impact:**  
  Observation: The campaign included file encryption and placement of a
  ransom note (!\_READ_ME_REXX2\_!.txt), consistent with ransomware
  impact techniques.

 

















