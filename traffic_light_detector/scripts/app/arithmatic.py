class Project:
    label_sample=[  
        {
            'type':'Green-up',
            'cordinates':{
                "xmin":100,
                "ymin":100,
                "xmax":1000,
                "ymax":1000
            }
        },
        {
            'type':'Red',
            'cordinates':{
                "xmin":100,
                "ymin":200,
                "xmax":1500,
                "ymax":1600
            }
        },
        {
            'type':'Yellow',
            'cordinates':{
                "xmin":100,
                "ymin":100,
                "xmax":1000,
                "ymax":1000
            }
        }
    ]
    def __init__(self, project, phase):
        self.project = project
        self.phase = phase
        # self.label_sample=label_sample

    def myfunc(self):
        return "Hello this is " + self.project+" phase: "+self.phase

    # def label_sample_creation(self):
    #     return self.label_sample
