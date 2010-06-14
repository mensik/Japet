#! /usr/bin/env python

import subprocess, ConfigParser

print "***********************************"

config = ConfigParser.SafeConfigParser()
config.read('default.cfg')

programs = ['testFeti', 'testSmale']
i = 0

for p in programs:
	print i,":", p, "\t",
	i = i + 1
try:
	i = int(raw_input("\nSelect program: (0) "))
except ValueError:
	i = 0;
try:
	selectedProgram = programs[i]
except IndexError:
	selectedProgram = programs[0]
	print "Ilegal selection. Default value (", selectedProgram ,") chosen."

x = ["Number of domain x","grid","x","-test_x", int]
y = ["Number of domain y","grid","y","-test_y", int]
h = ["Grid step","grid","h","-test_h", float]
d_count = ["Number of Dirchlet bounds","conditions","dirchlet_side_count","-test_bounded_side_count", int]
f = ["Select force function","conditions","force_function","-test_f", int]
args = [x,y,h,f,d_count]

for arg in args:
	try:
		if arg[4] == int:
			arg[5:5] = [config.getint(arg[1],arg[2])]
		elif arg[4] == float:
			arg[5:5] = [config.getfloat(arg[1],arg[2])]
		
		arg[5] = arg[4](raw_input(arg[0] + ": (" + str(arg[5]) + ") "))
	except ValueError:
		pass	

carr = ["mpiexec", "-n", str(args[0][5] * args[1][5]), "./" + selectedProgram]
for arg in args:
	carr = carr + [arg[3], str(arg[5])]
	config.set(arg[1],arg[2],str(arg[5]))

with open('default.cfg','w') as f:
	config.write(f)

print "***********************************"

subprocess.call(carr)
