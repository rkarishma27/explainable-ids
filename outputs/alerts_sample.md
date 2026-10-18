# Sample alert cards (seed 0 test set, random sample)

**Flow 22034**  136.224.67.69 -> 10.3.123.224  |  **PortScan**  (confidence 100%)

- Why: inter-arrival time std-dev (us) = 0 (+3.99); initiator TCP window (B) = 5840 (+2.99); flow duration (us) = 2345.67 (+1.69)
- Argues against: packet-length variance = 0 (-0.08); mean inter-arrival time (us) = 2345.67 (-0.06)
- Ground truth (synthetic/test label): PortScan

**Flow 28663**  129.55.129.152 -> 10.2.54.177  |  **PortScan**  (confidence 100%)

- Why: inter-arrival time std-dev (us) = 0 (+4.76); flow duration (us) = 898.47 (+3.26); packets sent by initiator = 1 (+0.48)
- Argues against: mean active time (us) = 898.47 (-0.21); mean initiator inter-arrival (us) = 898.47 (-0.08)
- Ground truth (synthetic/test label): PortScan

**Flow 26220**  35.85.9.231 -> 10.1.4.60  |  **BruteForce**  (confidence 100%)

- Why: destination port = 22 (+5.13); mean idle time (us) = 0 (+2.61); inter-arrival time std-dev (us) = 20.9k (+0.97)
- Argues against: ACK flags = 54 (-0.24); responder/initiator packet ratio = 0 (-0.07)
- Ground truth (synthetic/test label): BruteForce

**Flow 24652**  60.135.247.218 -> 10.1.117.184  |  **Botnet**  (confidence 90%)

- Why: inter-arrival time std-dev (us) = 9.5M (+3.56); mean idle time (us) = 13.8M (+1.86); bytes sent by responder = 1732.54 (+1.13)
- Argues against: mean initiator inter-arrival (us) = 7.7M (-0.72); packets sent by responder = 7 (-0.42)
- Ground truth (synthetic/test label): Botnet

**Flow 2229**  142.110.2.92 -> 10.3.72.6  |  **DoS**  (confidence 100%)

- Why: SYN flags = 4 (+3.65); mean initiator packet size (B) = 11.44 (+1.23); PSH flags = 2 (+0.59)
- Argues against: mean idle time (us) = 0 (-0.15); mean responder packet size (B) = 138.04 (-0.05)
- Ground truth (synthetic/test label): DoS

**Flow 5606**  59.140.20.6 -> 10.1.228.119  |  **DoS**  (confidence 99%)

- Why: SYN flags = 2 (+4.11); byte rate (B/s) = 0 (+0.91); destination port = 80 (+0.63)
- Argues against: packets sent by initiator = 2 (-0.35); mean idle time (us) = 0 (-0.18)
- Ground truth (synthetic/test label): DoS

**Flow 22440**  128.140.73.172 -> 10.3.26.181  |  **Botnet**  (confidence 94%)

- Why: inter-arrival time std-dev (us) = 22.0M (+5.06); mean active time (us) = 362.8k (+0.62); PSH flags = 5 (+0.50)
- Argues against: packet rate (pkt/s) = 0.24 (-0.33); mean inter-arrival time (us) = 4.5M (-0.29)
- Ground truth (synthetic/test label): Botnet

**Flow 32537**  102.132.72.227 -> 10.3.140.143  |  **DoS**  (confidence 100%)

- Why: SYN flags = 2 (+4.36); initiator TCP window (B) = 8192 (+0.89); byte rate (B/s) = 0 (+0.78)
- Argues against: packets sent by initiator = 2 (-0.32); mean idle time (us) = 0 (-0.18)
- Ground truth (synthetic/test label): DoS

**Flow 15462**  28.205.136.212 -> 10.0.93.113  |  **DoS**  (confidence 96%)

- Why: mean initiator packet size (B) = 13.33 (+2.84); PSH flags = 3 (+0.79); inter-arrival time std-dev (us) = 206.2k (+0.75)
- Argues against: FIN flags = 1 (-0.38); initiator TCP window (B) = 65.5k (-0.19)
- Ground truth (synthetic/test label): DoS

**Flow 17957**  123.129.79.226 -> 10.1.240.49  |  **BruteForce**  (confidence 100%)

- Why: destination port = 22 (+5.17); mean idle time (us) = 0 (+2.56); inter-arrival time std-dev (us) = 79.1k (+0.94)
- Argues against: mean initiator packet size (B) = 70.53 (-0.26); responder/initiator packet ratio = 0 (-0.05)
- Ground truth (synthetic/test label): BruteForce

**Flow 6989**  195.204.10.26 -> 10.0.29.17  |  **DoS**  (confidence 99%)

- Why: SYN flags = 2 (+4.02); byte rate (B/s) = 0 (+0.98); PSH flags = 0 (+0.52)
- Argues against: packets sent by initiator = 2 (-0.38); mean idle time (us) = 0 (-0.17)
- Ground truth (synthetic/test label): DoS

**Flow 22876**  128.86.107.95 -> 10.1.208.35  |  **PortScan**  (confidence 98%)

- Why: destination port = 61.3k (+4.52); inter-arrival time std-dev (us) = 0 (+3.28); byte rate (B/s) = 0 (+0.46)
- Argues against: flow duration (us) = 158.0k (-0.19); packet-length variance = 0 (-0.13)
- Ground truth (synthetic/test label): PortScan

**Flow 9314**  10.0.116.245 -> 10.2.238.140  |  **WebAttack**  (confidence 60%)

- Why: mean initiator packet size (B) = 298.58 (+3.14); packets sent by initiator = 36 (+1.01); destination port = 80 (+0.58)
- Argues against: mean responder packet size (B) = 162.75 (-0.13); initiator TCP window (B) = 65.5k (-0.05)
- Ground truth (synthetic/test label): BENIGN

**Flow 30924**  10.3.33.175 -> 10.2.189.153  |  **WebAttack**  (confidence 63%)

- Why: mean initiator packet size (B) = 210.99 (+3.42); PSH flags = 18 (+2.37); bytes sent by initiator = 10.3k (+0.39)
- Argues against: packets sent by initiator = 49 (-0.18); mean responder packet size (B) = 238.82 (-0.14)
- Ground truth (synthetic/test label): BENIGN

**Flow 28940**  10.1.173.27 -> 10.0.93.198  |  **Botnet**  (confidence 59%)

- Why: inter-arrival time std-dev (us) = 43.7M (+4.32); mean idle time (us) = 11.7M (+1.41); packets sent by initiator = 5 (+0.49)
- Argues against: mean initiator packet size (B) = 58.71 (-0.34); packet rate (pkt/s) = 0.12 (-0.29)
- Ground truth (synthetic/test label): BENIGN

