#!/bin/env python2 
# coding: utf-8
import dsf,os,numpy,pickle
from numpy import array,loadtxt,savetxt,zeros,sort,dtype
from numpy import float32,float64,real,imag,exp
import subprocess
from dsf import trajectory_readers

#TODO get initial parameters with argparse module
#TODO select kvectors from from scalar

"""example for non-linear trajectories periodicity=weed"""
path="/tmpmnt/home/titanic3/sfrey/LJ_800x10/system-A/NVT-system-A/T0.43/ana-trajectory/"
name="mdAVT0.43"
maxframe=170
postype=".poslog.trjlog"
veltype=".vellog.trjlog"
weed=34

"""example for linear trajectories periodicity=1 frame"""
#path="/tmpmnt/home/titanic3/sfrey/LJ_800x10/system-B/NVT-system-B/T0.43-2part/ana-trajectory/"
#ii=starttraj=100
#ij=stoptraj=601
#name="mdBVT0.43"
#maxframe=100
#postype=".pos.trj"
#veltype=".vel.trj"
#weed=1

"""init var"""
k=1
ACL=[]
ACT=[]

def complex_to_line(CA):
    """
    CA mean complex array with dim=1
    return one line with even elem are real and odd complex
    array[ C1 , C2 , ... ,Cn ] -> ' real(C1) imag(C1) .... imag(cn)'
    """
    fl=lambda x: " "+str(real(x))+" "+str(imag(x))
    return reduce(str.__add__, map(fl,CA) )

ytPath='/usr/local/opt/yasp03/bin/'
tool='calc_cfav'
options=" -ncol %i -c 2 -mfft -1  -m 1000000 "%(6+12+1)
pinput,poutput=os.popen2(ytPath+tool+options)

#options=[ " -ncol %i "%(6+1),"-c 2 ","-mfft -1 "," -m 1000000 " ]
#PIPE=subprocess.PIPE
#args=[ytPath+tool]+options
#print args
#process=subprocess.Popen(args, bufsize=0, stdout=PIPE , stdin=PIPE)
#pinput=process.stdin
#poutput=process.stdout
""" subprocess version for python 3"""
def Jflux(v,p):
    """
    Mostly compute permutation
    """
    CL=[(v[0]*p[0]).sum(),(v[1]*p[1]).sum(),(v[2]*p[2]).sum()]
    CT=[(v[0]*p[1]).sum(),(v[1]*p[2]).sum(),(v[2]*p[0]).sum(),
        (v[0]*p[2]).sum(),(v[1]*p[0]).sum(),(v[2]*p[1]).sum()]
    return array(CL),array(CT)
"""
nbst = numbert for short time analysis done ( sum/nbst = avg )
CLST = accumate value of CT coorelation for short time
CLZW = CL for last value of weed start ( reference t=0 )
"""    
frame=0
nbst=0
CLST={}
CTST={}
timev={}
f=open('testout.txt','w')
for ik in range(ii,ij):
    print 'working on trajectory %i/%i at frame %i'%(ik+1-ii,ij-ii,ik)
    pname=path+name+"."+str(ik)+postype
    vname=path+name+"."+str(ik)+veltype
    p=trajectory_readers.molfile_reader(pname)
    v=trajectory_readers.molfile_reader(vname)
    for il in range(maxframe):
        pos=p.next()
        vel=v.next()
        if (pos['time'] != vel['time']):
            print 'time for position and velocity are misalign' 
            raise Error
        time=float64(pos['time'])
        N=len(pos['x'][0])
        if (frame%weed == 0):
            nbst+=1
            twz=time
            CL,CT=Jflux(10*vel['x'],exp((0+1j)*k+10*pos['x']))
            CL,CT=CL/N,CT/N
            CLZW=CL
            CTZW=CT
            ligne=str(time)
            #print ligne
            ligne+=complex_to_line(CL)+complex_to_line(CT)
            pinput.write(ligne+'\n')
            f.write(ligne+'\n')
            idt=-1
        else:
            CL,CT=Jflux(10*vel['x'],exp((0+1j)*k+10*pos['x']))
            CL,CT=CL/N,CT/N
            idt+=1
            if not CTST.has_key(idt):
                print "init %i at time %f to %f "%(idt,twz,time)
                timev[idt]=time-twz
                CTST[idt]=0
                CLST[idt]=0
            CTST[idt]+=(real(CT)*real(CTZW)+imag(CT)*imag(CTZW)).mean()
            CLST[idt]+=(real(CL)*real(CLZW)+imag(CL)*imag(CLZW)).mean()
            #print dt,CLST[dt],(real(CT)*real(CTZW)+imag(CT)*imag(CTZW))/N**2
        frame+=1
        # do not store value , pipe to tools
        #ACL.append(CL)
        #ACT.append(CT)
    p.close()
    v.close()
f.close()
pinput.close()
buff=poutput.readlines()
poutput.close()
f=open('CTbuff.pickle','w')
pickle.dump(buff,f)
from StringIO import StringIO
# list of string -> string -> array((t,18)) 
res=loadtxt(StringIO(reduce(str.__add__,buff))).T
fres=zeros((3,res.shape[1]))
fres[0]=res[0]
fres[1]=2*res[1:7].mean(axis=0)
fres[2]=2*res[7:19].mean(axis=0)
fres=fres.T
resDtype=dtype([('time','f'),('CL','f'),('CT','f')])
if weed > 1:
    keyl=sort(CLST.keys())
    resSt=array([ [timev[key],CLST[key]/nbst,CTST[key]/nbst]  for key in keyl ])
else:
    resSt=[]
resTot=array([tuple(l) for l in resSt]+[tuple(l) for l in fres[1:] ],resDtype)
pickle.dump(resTot,f)
f.close()
f=open('CT-%i-%i.txt'%(ii,ij),'w')
f.write('#this CL[6]/N+CT[12]/N autocorrelation for k=%f\n'%k)
f.write('#autorrelation done with %s\n'%(ytPath+tool+options))
f.write('#for frame %i to %i with %i vonf by frame \n'%(ii,ij,maxframe))
f.write('# t CL CT \n')
f.write('#Path to sources %s*\n'%(path+name))
savetxt(f,resTot)
f.write('#end file\n')
f.close()
