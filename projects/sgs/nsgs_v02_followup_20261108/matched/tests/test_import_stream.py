import importlib.util,sys,tempfile,threading,time,unittest
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'import_stream.py'
spec=importlib.util.spec_from_file_location('import_stream',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class StreamImportTests(unittest.TestCase):
    def test_growing_file_not_premature_eof(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'archive.partial';data=bytes(range(256))*4;p.write_bytes(data[:16])
            original=module.EXPECTED_BYTES;module.EXPECTED_BYTES=len(data)
            try:
                reader=module.GrowingReader(p)
                def append():
                    time.sleep(.03)
                    with p.open('ab') as f:f.write(data[16:])
                worker=threading.Thread(target=append);worker.start()
                self.assertEqual(reader.read(len(data)),data);worker.join()
                self.assertEqual(reader.read(1),b'');reader.f.close()
            finally:module.EXPECTED_BYTES=original
    def test_incomplete_archive_times_out(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'archive.partial';p.write_bytes(b'')
            reader=module.GrowingReader(p);reader.deadline=time.monotonic()-1
            with self.assertRaises(TimeoutError):reader.read(1)
            reader.f.close()
if __name__=='__main__':unittest.main()
