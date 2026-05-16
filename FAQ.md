- When analyzing videos or images, are any files uploaded to an external server, cloud
  platform, or third-party data center?

  Dynaface is not designed to transmit data to external servers. To the best of our
  knowledge and based on current design, all image and video processing occurs locally
  on the user's machine.

- Does Dynaface store any image/video data, metadata, or analysis outputs outside of
  the local machine?

  Dynaface is not designed to store data outside of the local machine. No functionality
  within the application is intended to write data to remote locations.

- Are there any dependencies or features that require internet access during analysis?

  Dynaface does not perform internet access during image or video analysis; all
  processing is performed locally. The only internet access initiated by Dynaface is
  to open an external web browser to view the online user manual
  (https://github.com/jeffheaton/dynaface/blob/main/dynaface-app/manual.md). Users
  should be aware that standard browser telemetry or network activity from their
  browser is outside the scope of Dynaface.

- Are outputs limited to local files such as CSV/Excel exports, or is any data
  transmitted externally?

  All output produced by Dynaface is written to locations on the local file system as
  specified by the user. No output is transmitted externally by the application.

- Are there any specific software, licensing, or data-use considerations we should be
  aware of for academic clinical research use?

  Dynaface is released under the Apache 2.0 open-source license
  (https://github.com/jeffheaton/dynaface/blob/main/LICENSE). Researchers should
  review the license terms directly, as they govern permitted uses, redistribution,
  and disclaimers of warranty. No additional proprietary licensing is imposed by the
  Dynaface project itself. Institutional review or compliance requirements for clinical
  research (such as IRB approval or HIPAA considerations) are the responsibility of
  the researcher's institution and are not addressed by the Dynaface license.
