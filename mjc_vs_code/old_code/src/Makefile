OBJDIR = ../obj
BINDIR = ../bin
LIBDIR = ../lib
CYGWINRM = C:/cygwin64/bin/rm.exe

WINCPP = cl
WINCOMMON=/O2 /MT /EHsc /arch:AVX /I../include
WINMUJOCOFLAGS = $(LIBDIR)/mujoco.lib
WINOPENGL = $(LIBDIR)/glfw3.lib
WINNLOPT = $(LIBDIR)/libnlopt.lib


UITOOLS = ../include/uitools.c

simulate:
	$(WINCPP) $(WINCOMMON) simulate.cpp /Fe$(BINDIR)/simulate.exe /Fo$(OBJDIR)/ $(UITOOLS) $(WINOPENGL) $(WINMUJOCOFLAGS) $(WINNLOPT)

simulate_minimize:
	$(WINCPP) $(WINCOMMON) simulate_minimize.cpp /Fe$(BINDIR)/simulate_minimize.exe /Fo$(OBJDIR)/ $(UITOOLS) $(WINOPENGL) $(WINMUJOCOFLAGS) $(WINNLOPT)

simulate_cylinder:
	$(WINCPP) $(WINCOMMON) simulate_cylinder.cpp /Fe$(BINDIR)/simulate_cylinder.exe /Fo$(OBJDIR)/ $(UITOOLS) $(WINOPENGL) $(WINMUJOCOFLAGS) $(WINNLOPT)

simulate_grab_torques:
	$(WINCPP) $(WINCOMMON) simulate_grab_torques.cpp /Fe$(BINDIR)/simulate_grab_torques.exe /Fo$(OBJDIR)/ $(UITOOLS) $(WINOPENGL) $(WINMUJOCOFLAGS) $(WINNLOPT)

simulate_grab_compliance:
	$(WINCPP) $(WINCOMMON) simulate_grab_compliance.cpp /Fo$(OBJDIR)/ /Fe$(BINDIR)/simulate_grab_compliance.exe $(UITOOLS) $(WINOPENGL) $(WINMUJOCOFLAGS) $(WINNLOPT)

sgt: simulate_grab_torques

sgc: simulate_grab_compliance

all: simulate simulate_minimize simulate_cylinder

clean:
	$(CYGWINRM) -f $(OBJDIR)/*.obj *.obj *.exp *.lib *.mexw64.manifest *.dll
