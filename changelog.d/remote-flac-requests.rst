**Read remote FLAC files with fewer requests**
  Remote files read in small pieces, such as FLAC signal files on PhysioNet, now use larger requests as reading goes on (up to 8 MB) instead of one request for every 32 kB. Reading a 7.4 MB MIMIC-IV signal file now takes 8 requests instead of 229, which saves several minutes per file.
