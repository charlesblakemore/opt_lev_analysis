import numpy as np
import xmltodict


test_file = '/daq2/20190306/test4/test.attr'

xml = open(test_file, 'r').read()

attr_dict = xmltodict.parse(xml)['Cluster']

n_attr = int(attr_dict['NumElts'])

types = ['DBL', 'Array', 'Boolean', 'String']

new_attr_dict = {}
for attr_type in types:
    c_list = attr_dict[attr_type]
    if type(c_list) != list:
        c_list = [c_list]

    for item in c_list:
        new_key = item['Name']

        # Keep the time as 64 bit unsigned integer
        if new_key == 'Time':
            new_attr_dict[new_key] = np.uint64(float(item['Val']))
        
        # Convert single numbers/bool from their xml string representation
        elif (attr_type == 'DBL') or (attr_type == 'Boolean'):
            new_attr_dict[new_key] = float(item['Val'])

        # Convert arrays of numbers from their parsed xml
        elif (attr_type == 'Array'):
            new_arr = []
            vals = item['DBL']
            for val in vals:
                new_arr.append(float(val['Val']))
            new_attr_dict[new_key] = new_arr

        # Move string attributes to new attribute dictionary
        elif (attr_type == 'String'):
            new_attr_dict[new_key] = item['Val']

print new_attr_dict
            
    
    
             
