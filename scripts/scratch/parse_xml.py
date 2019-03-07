import numpy as np
import xmltodict


test_file = '/daq2/20190306/test4/test.attr'

xml = open(test_file, 'r').read()

attr_dict = xmltodict.parse(xml)['Cluster']

n_attr = int(attr_dict['NumElts']

types = ['DBL', 'Array', 'Boolean', 'String']

for attr_typ in types:
    print type
    
    
    
