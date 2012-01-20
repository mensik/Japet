#! /usr/bin/env python

import ConfigParser

f = open('configurations','r')
conFiles = f.readlines()
f.close()

print "\n******************************"
print "   Japet Config generator"
print "******************************\n"
for i in range(len(conFiles)):
	conFiles[i] = conFiles[i].strip()
	print str(i) + ")  " + conFiles[i]

try:
	i = int(raw_input("\nSelect base configuration: (0) "))
except ValueError:
	i = 0;
try:
	confFile = conFiles[i]
except IndexError:
	confFile = conFiles[0]
	print "Ilegal selection. Default value was (", confFile ,") chosen."

config = ConfigParser.SafeConfigParser()
config.read(confFile)

while 1:
	print "\n"
	values = config.items('feti')
	for i in range(len(values)):
		print str(i + 1) + ") " + values[i][0] + " = " + values[i][1]

	try:
		i = int(raw_input("\nSelect value to change (0 - continue) : "))
	except ValueError:
		i = 0;

	if i == 0:
		break

	try:
		newVal = str(raw_input("New value for '" + values[i-1][0] + "' : "))
		config.set('feti',values[i-1][0],	newVal)
	except IndexError:
		pass


scriptName = str(raw_input("Script name : "))
f = open(scriptName, 'w')

values = config.items('feti')

m = config.getint('feti','japet_m')
n = config.getint('feti','japet_n')
subDomains = m * n

f.write("mpiexec -n " + str(subDomains) + " ./testFeti")
for val in values:
	f.write(" -" + val[0] + " " + val[1])

f.close

q = str(raw_input('Save config ? '))

if q == 'y' :
	cName = str(raw_input('Config name '))
	f = open(cName, 'w')
	config.write(f)
	f.close()

	f = open('configurations','a')
	f.write(cName + "\n")
	f.close()
