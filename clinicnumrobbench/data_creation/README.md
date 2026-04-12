To access MIMIC III, MIMIC IV, MedNLI, you must fulfill all of the following requirements:
- be a [credentialed user](https://physionet.org/settings/credentialing/)
- complete required training: [CITI Data or Specimens Only Research](https://physionet.org/about/citi-course/). You may submit your training [here](https://physionet.org/settings/training/).
- sign [the data use agreement](https://physionet.org/sign-dua/mimiciv/3.1/) for the project

## Download
You currently at root folder: ClinicNumRobBench/

1.Create data folder: `mkdir -p data/physionet`
2. move to data folder: `cd data/physionet`
3. Download
I remmend to create `tmux` or `smux` session to avoid interruption due to long time downloading

You need to input your physionet password upon running these commands
- set username: `export USERNAME=`
- mimic iv 3.1: `wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/mimiciv/3.1/`
  >> Downloaded: 38 files, 9.9G in 13h 38m 24s (212 KB/s)
- mimiciv4ed: `wget -r -N -c -np --user $USERNAME --ask-password https://physionet.org/files/mimic-iv-ed/2.2/`
  >> Downloaded: 11 files, 67M in 1h 40m 38s (11.4 KB/s)

Data Structure should be
```
data/physionet/physionet.org/files
├── mimic-iv-ed
│   ├── 2.2
│   │   ├── ed
│   │   │   ├── vitalsign.csv.gz
│   │   │   ├── ...
│   │   ├── LICENSE.txt
├── mimiciv
│   ├── 3.1
│   │   ├── hosp
│   │   │   ├── patients.csv.gz
│   │   │   ├── ...
│   │   ├── icu
│   │   │   ├── inputevents.csv.gz
│   │   │   ├── icustays.csv.gz
│   │   │   ├── ...
│   │   ├── LICENSE.txt
```
