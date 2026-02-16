## CongressText

This repository is built from [`JMSLab/Template`](https://github.com/JMSLab/Template/tree/d39df7414aad471b3df214f6684191496b2a90fd) and by default shares its dependencies and prerequisites.

The datastore is [`CongressText`](https://drive.google.com/drive/u/1/folders/0ALDmSBTLfB78Uk9PVA).

### Workflow summary
This folder produces text data containing the speeches contained in the Daily Congressional Record. To accomplish this, we use the following procedure:
1. Scrape the Congress.gov website for HTML and PDF scans of the Daily Congressional Record
2. Use the [congressional-record](https://github.com/unitedstates/congressional-record/tree/main) repository to parse the Congressional Record to JSON
3. Convert the JSON to our historical schema
