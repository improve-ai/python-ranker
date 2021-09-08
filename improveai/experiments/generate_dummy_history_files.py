import os

from coolname import generate
from datetime import datetime
import hashlib
import gzip
import json
import numpy as np
from tqdm import tqdm
from uuid import uuid4


history_record_dummy = \
    {"timestamp": None,  # "2021-09-02T01:29:07.301Z",
     "message_id": None,  # "7cadd313-93ea-4098-8321-f09424d9f947",
     "type": None,  # "decision",
     "model": None,  # "appconfig",
     "variant": None,  # 0,
     "count": None,  # 1,
     "givens": None,  # {},
     "received_at": None}  # "2021-09-01T23:29:07.924Z"}

if __name__ == '__main__':

    hist_count = 20
    models = ['appconfig', 'songs-2.0', 'themes-2.0', 'images-2.0', 'dummy-2.0']

    histories_dir = '/home/kw/Projects/upwork/sanity_check_artifacts/incoming'

    os.system('rm -rf {}'.format(histories_dir))
    os.system('mkdir -p {}'.format(histories_dir))

    for _ in tqdm(range(hist_count)):

        history = '_'.join(generate(2))
        history_hash = hashlib.sha256(history.encode()).hexdigest()
        model = np.random.choice(models)

        for _ in range(np.random.randint(6, 9)):

            records = []
            for _ in range(np.random.randint(20, 45)):

                record_type = np.random.choice(['decision'] * 4 + ['event'])

                ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

                variant_text = '_'.join(generate(3))

                variant = {
                    'text': variant_text,
                    'text_len': len(variant_text)}

                givens = {}
                if np.random.rand() >= 0.8:
                    givens = {
                        'g1': np.random.randint(0, 100),
                        'g2': np.random.rand()}

                if record_type == 'decision':
                    history_record = \
                        {"timestamp": ts,
                         "message_id": str(uuid4()),  # "7cadd313-93ea-4098-8321-f09424d9f947",
                         "type": record_type,
                         "model": model,  # "appconfig",
                         "variant": variant,  # 0,
                         "count": 1,  # 1,
                         "givens": givens,  #{},
                         "received_at": ts}
                else:
                    history_record = \
                        {"timestamp": ts,
                         "message_id": str(uuid4()),  # "7cadd313-93ea-4098-8321-f09424d9f947",
                         "type": record_type,
                         "model": model,  # "appconfig",
                         "properties": {"value": np.random.rand()},  # 0,
                         "count": 1,  # 1,
                         "received_at": ts}

                records.append(history_record)

            lines = '\n'.join([json.dumps(r) for r in records]).encode()
            gzipped_lines = gzip.compress(lines)

            curr_history_uuid = str(uuid4())
            curr_history_filename = \
                '{}-{}.jsonl.gz'.format(history_hash, curr_history_uuid)

            curr_file_path = '{}/{}'.format(histories_dir, curr_history_filename)

            with open(curr_file_path, 'wb') as gzf:
                gzf.write(gzipped_lines)
