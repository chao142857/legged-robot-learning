"""Inspect the executed strict-success protocol without launching a simulator."""
import argparse,json
from nsgs.common import PROTOCOL
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--protocol',action='store_true',help='Print the original evaluation protocol')
a=p.parse_args()
if a.protocol: print(json.dumps(PROTOCOL,indent=2))
else: p.print_help()
