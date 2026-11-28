"""CPU fixture checks for real-smoke reference selection; no neural execution."""
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/colab_server'))
from live_smoke_inputs import prepare_smoke_inputs

class SmokeInputs(unittest.TestCase):
    def test_cutout_has_only_visible_kurta_and_rejects_changed_mask(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'05_SPLITS').mkdir()
            image=root/'donor.png';mask=root/'mask.png';Image.new('RGB',(6,8),'red').save(image)
            labels=np.zeros((8,6),dtype=np.uint8);labels[2:6,1:5]=1;labels[3,2]=3
            Image.fromarray(labels).save(mask)
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            def write(p,rows):
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            c=dict(case_id='C001',scenario='adult_to_adult',source_image_id='d',target_image_id='p',mode='upper_body')
            write(root/'05_SPLITS/cases.csv',[c])
            write(root/'05_SPLITS/frozen.csv',[dict(image_id=i,image_path=str(image),mask_path=str(mask),mask_sha256=sha(mask),split='test') for i in ['d','p']])
            folder=root/'10_RUNS/benchmark_baseline_fashn_vton_1_5';folder.mkdir(parents=True)
            (folder/'protocol.json').write_text(json.dumps(dict(cases=[c],inputs={i:dict(image=sha(image)) for i in ['d','p']})))
            result=prepare_smoke_inputs(root,root/'out')
            with Image.open(result['catvton_reference']) as im:out=np.asarray(im)
            self.assertTrue(np.all(out[labels!=1]==255));self.assertTrue(np.all(out[labels==1]==[255,0,0]))
            Image.new('L',(6,8),1).save(mask)
            with self.assertRaisesRegex(ValueError,'mask changed'):prepare_smoke_inputs(root,root/'out')

if __name__=='__main__':unittest.main()
