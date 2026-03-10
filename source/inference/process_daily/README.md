### Daily Congressional Record

We build on code from the [congressional-record](https://github.com/unitedstates/congressional-record/) repository. We first parse the daily CR using the `congressional-record` schema: `submit_jobs_parse_daily.sh`

We then adapt the schema so that it matches our historical congressional record data schema: `submit_jobs_daily_to_historical_schema.sh`

Modification of the original code is permitted under the [original license](https://github.com/unitedstates/congressional-record/blob/ec0850412acffd0fc2dc6ed79fbd376e4699439d/LICENSE).

