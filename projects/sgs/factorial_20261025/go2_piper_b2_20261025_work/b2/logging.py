"""Add frozen run/support context to the owner's JSONL events."""
import json

class ContextLog:
    def __init__(self,path,context):
        self.context=dict(context)
        self.file=open(path,'x',buffering=1)
    def write(self,line):
        row=json.loads(line)
        for key,value in self.context.items():
            if key in row and row[key]!=value:raise ValueError('Cross-context event: '+key)
        return self.file.write(json.dumps(dict(self.context,**row),allow_nan=False)+'\n')
    def close(self):self.file.close()
