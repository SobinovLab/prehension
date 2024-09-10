/*  Copyright © 2018, Roboti LLC

    This file is licensed under the MuJoCo Resource License (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.roboti.us/resourcelicense.txt
*/


#include "mjxmacro.h"
#include "uitools.h"
#include "stdio.h"
#include "string.h"

#include <thread>
#include <mutex>
#include <chrono>

#include <math.h>
#include <iterator>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <algorithm>
#include <limits>

#include <nlopt.h>

// MY
#define HAND_TOUCHPAD_DISTANCE_CAP              10.0
#define COST_FUNCTION_CONTACTMAP_WEIGHT         1.
#define COST_FUNCTION_DISTANCE_WEIGHT           10.
#define COST_FUNCTION_PENETRATION_WEIGHT        1000.
#define COST_FUNCTION_FINGERTIPS_WEIGHT         10.
#define COST_FUNCTION_MAGNETISM_WEIGHT          0.1

#define COST_FUNCTION_LOCAL_EVAL                2000
#define COST_FUNCTION_GLOBAL_EVAL               500000

//-------------------------------- global -----------------------------------------------

// constants
const int maxgeom = 10000;           // preallocated geom array in mjvScene
const double syncmisalign = 0.1;    // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7;   // fraction of refresh available for simulation


// model and data
mjModel* m = NULL;
mjData* d = NULL;
char filename[1000] = "";


//----------------------------- MY
// body-geom names
const std::vector<std::string> HAND_SEGMENT_PREFIXES = {"RA3", "RA4", "RA5", "RA6"};
const std::string TOUCHPAD_BODY_NAME_PREFIX =           "TouchPad3D";
const std::string TOUCHPAD_MANIPULATOR_JOINT_PREFIX =   "ra_tp";
const std::string TOUCHPAD_CENTREISH_GEOM =             "TouchPad3D_09_05_geom";
const std::string HAND_FINGER_GEOM =                    "RA6D3_geom";
const std::string TOUCHPAD_RESISTIVE_SEGMENT =          "RATPR";
const std::vector<std::string> HAND_FINGERTIP_GEOMS =   {"RA6D2_geom", "RA6D3_geom", "RA6D4_geom", "RA6D5_geom"};
const std::string TOUCHPAD_BASE_GEOM =                  "RATP5_geom";

int desired_num_fingertip_contacts  = 4;    // can be extracted directly from the touchpad

// runtime updated cost function variables
float runtime_hand_touchpad_distance = HAND_TOUCHPAD_DISTANCE_CAP;
float runtime_metric = -1;
float runtime_contact_diff = -1;
float runtime_resseg_hand_penetration = -1;
int runtime_num_fingertips_touchpad = -1;
float runtime_fingertips_touchpad_distance = HAND_TOUCHPAD_DISTANCE_CAP;
bool runtime_hand_is_touching = false;
float runtime_touchpad_hand_magnetism = -1;

// static model-dependant variables
// float initial_contact_diff = -1;
float* desired_contact_map = NULL;
int num_touchpads = 0;
int* touchpad_geom_ids = NULL;
int num_touchpad_manipulator_joints = 0;
int* touchpad_manipulator_joints = NULL;
int num_hand_segs = 0;
int* hand_seg_geom_ids = NULL;
int resistive_segment_geom_id = 0;
int touchpad_base_geom_id = -1;
std::vector<int> hand_fingertip_geom_ids;

// cost function storage variables
int how_many_times_you_called_me = 0;
int iteration_when_min_was_found = -1;
float min_cf_val = std::numeric_limits<float>::max();
mjtNum *min_cf_pose = NULL;

// interface
int print_current_hand_touchpad_contacts = 0;


//----------------------------- ENDMY
// abstract visualization
mjvScene scn;
mjvCamera cam;
mjvOption vopt;
mjvPerturb pert;
mjvFigure figconstraint;
mjvFigure figcost;
mjvFigure figtimer;
mjvFigure figsize;
mjvFigure figsensor;


// OpenGL rendering and UI
GLFWvidmode vmode;
int windowpos[2];
int windowsize[2];
mjrContext con;
GLFWwindow* window = NULL;
mjuiState uistate;
mjUI ui0, ui1;


// UI settings not contained in MuJoCo structures
struct
{
    // file
    int exitrequest = 0;

    // option
    int spacing = 0;
    int color = 0;
    int font = 0;
    int ui0 = 1;
    int ui1 = 1;
    int help = 0;
    int info = 0;
    int profiler = 0;
    int sensor = 0;
    int fullscreen = 0;
    int vsync = 1;
    int busywait = 0;

    // simulation
    int run = 1;
    int key = 0;
    int loadrequest = 0;

    // watch
    char field[mjMAXUITEXT] = "qpos";
    int index = 0;

    // physics: need sync
    int disable[mjNDISABLE];
    int enable[mjNENABLE];

    // rendering: need sync
    int camera = 0;
} settings;


// section ids
enum
{
    // left ui
    SECT_FILE   = 0,
    SECT_OPTION,
    SECT_SIMULATION,
    SECT_WATCH,
	SECT_PHYSICS,
    SECT_RENDERING,
    SECT_GROUP,
    NSECT0,

    // right ui
    SECT_JOINT = 0,
    SECT_CONTROL,
    NSECT1
};


// file section of UI
const mjuiDef defFile[] =
{
    {mjITEM_SECTION,   "File",          1, NULL,                    "AF"},
    {mjITEM_BUTTON,    "Save xml",      2, NULL,                    ""},
    {mjITEM_BUTTON,    "Save mjb",      2, NULL,                    ""},
    {mjITEM_BUTTON,    "Print model",   2, NULL,                    "CM"},
    {mjITEM_BUTTON,    "Print data",    2, NULL,                    "CD"},
    {mjITEM_BUTTON,    "Quit",          1, NULL,                    "CQ"},
    {mjITEM_END}
};


// option section of UI
const mjuiDef defOption[] =
{
    {mjITEM_SECTION,   "Option",        1, NULL,                    "AO"},
    {mjITEM_SELECT,    "Spacing",       1, &settings.spacing,       "Tight\nWide"},
    {mjITEM_SELECT,    "Color",         1, &settings.color,         "Default\nOrange\nWhite\nBlack"},
    {mjITEM_SELECT,    "Font",          1, &settings.font,          "50 %\n100 %\n150 %\n200 %\n250 %\n300 %"},
    {mjITEM_CHECKINT,  "Left UI (Tab)", 1, &settings.ui0,           " #258"},
    {mjITEM_CHECKINT,  "Right UI",      1, &settings.ui1,           "S#258"},
    {mjITEM_CHECKINT,  "Help",          2, &settings.help,          " #290"},
    {mjITEM_CHECKINT,  "Info",          2, &settings.info,          " #291"},
    {mjITEM_CHECKINT,  "Profiler",      2, &settings.profiler,      " #292"},
    {mjITEM_CHECKINT,  "Sensor",        2, &settings.sensor,        " #293"},
#ifdef __APPLE__
    {mjITEM_CHECKINT,  "Fullscreen",    0, &settings.fullscreen,    " #294"},
#else
    {mjITEM_CHECKINT,  "Fullscreen",    1, &settings.fullscreen,    " #294"},
#endif
    {mjITEM_CHECKINT,  "Vertical Sync", 1, &settings.vsync,         " #295"},
    {mjITEM_CHECKINT,  "Busy Wait",     1, &settings.busywait,      " #296"},
    {mjITEM_END}
};


// simulation section of UI
const mjuiDef defSimulation[] =
{
    {mjITEM_SECTION,   "Simulation",    1, NULL,                    "AS"},
    {mjITEM_RADIO,     "",              2, &settings.run,           "Pause\nRun"},
    {mjITEM_BUTTON,    "Reset",         2, NULL,                    " #259"},
    {mjITEM_BUTTON,    "Reload",        2, NULL,                    "CL"},
    {mjITEM_BUTTON,    "Align",         2, NULL,                    "CA"},
    {mjITEM_BUTTON,    "Copy pose",     2, NULL,                    "CC"},
    {mjITEM_SLIDERINT, "Key",           3, &settings.key,           "0 0"},
    {mjITEM_BUTTON,    "Reset to key",  3},
    {mjITEM_BUTTON,    "Set key",       3},
    // {mjITEM_BUTTON,    "Print cont",    2, &print_current_hand_touchpad_contacts, "0 0"},
    {mjITEM_END}
};


// watch section of UI
const mjuiDef defWatch[] =
{
    {mjITEM_SECTION,   "Watch",         0, NULL,                    "AW"},
    {mjITEM_EDITTXT,   "Field",         2, settings.field,          "qpos"},
    {mjITEM_EDITINT,   "Index",         2, &settings.index,         "1"},
    {mjITEM_STATIC,    "Value",         2, NULL,                    " "},
    {mjITEM_END}
};


// help strings
const char help_content[] =
"Alt mouse button\n"
"UI right hold\n"
"UI title double-click\n"
"Space\n"
"Esc\n"
"Right arrow\n"
"Left arrow\n"
"Down arrow\n"
"Up arrow\n"
"Page Up\n"
"Double-click\n"
"Right double-click\n"
"Ctrl Right double-click\n"
"Scroll, middle drag\n"
"Left drag\n"
"[Shift] right drag\n"
"Ctrl [Shift] drag\n"
"Ctrl [Shift] right drag";

const char help_title[] =
"Swap left-right\n"
"Show UI shortcuts\n"
"Expand/collapse all  \n"
"Pause\n"
"Free camera\n"
"Step forward\n"
"Step back\n"
"Step forward 100\n"
"Step back 100\n"
"Select parent\n"
"Select\n"
"Center\n"
"Track camera\n"
"Zoom\n"
"View rotate\n"
"View translate\n"
"Object rotate\n"
"Object translate";


// info strings
char info_title[1000];
char info_content[1000];



//----------------------- profiler, sensor, info, watch ---------------------------------

// init profiler figures
void profilerinit(void)
{
    int i, n;

    // set figures to default
    mjv_defaultFigure(&figconstraint);
    mjv_defaultFigure(&figcost);
    mjv_defaultFigure(&figtimer);
    mjv_defaultFigure(&figsize);

    // titles
    strcpy(figconstraint.title, "Counts");
    strcpy(figcost.title, "Convergence (log 10)");
    strcpy(figsize.title, "Dimensions");
    strcpy(figtimer.title, "CPU time (msec)");

    // x-labels
    strcpy(figconstraint.xlabel, "Solver iteration");
    strcpy(figcost.xlabel, "Solver iteration");
    strcpy(figsize.xlabel, "Video frame");
    strcpy(figtimer.xlabel, "Video frame");

    // y-tick nubmer formats
    strcpy(figconstraint.yformat, "%.0f");
    strcpy(figcost.yformat, "%.1f");
    strcpy(figsize.yformat, "%.0f");
    strcpy(figtimer.yformat, "%.2f");

    // colors
    figconstraint.figurergba[0] =   0.1f;
    figcost.figurergba[2] =         0.2f;
    figsize.figurergba[0] =         0.1f;
    figtimer.figurergba[2] =        0.2f;
    figconstraint.figurergba[3] =   0.5f;
    figcost.figurergba[3] =         0.5f;
    figsize.figurergba[3] =         0.5f;
    figtimer.figurergba[3] =        0.5f;

    // legends
    strcpy(figconstraint.linename[0], "total");
    strcpy(figconstraint.linename[1], "active");
    strcpy(figconstraint.linename[2], "changed");
    strcpy(figconstraint.linename[3], "evals");
    strcpy(figconstraint.linename[4], "updates");
    strcpy(figcost.linename[0], "improvement");
    strcpy(figcost.linename[1], "gradient");
    strcpy(figcost.linename[2], "lineslope");
    strcpy(figsize.linename[0], "dof");
    strcpy(figsize.linename[1], "body");
    strcpy(figsize.linename[2], "constraint");
    strcpy(figsize.linename[3], "sqrt(nnz)");
    strcpy(figsize.linename[4], "contact");
    strcpy(figsize.linename[5], "iteration");
    strcpy(figtimer.linename[0], "total");
    strcpy(figtimer.linename[1], "collision");
    strcpy(figtimer.linename[2], "prepare");
    strcpy(figtimer.linename[3], "solve");
    strcpy(figtimer.linename[4], "other");

    // grid sizes
    figconstraint.gridsize[0] = 5;
    figconstraint.gridsize[1] = 5;
    figcost.gridsize[0] = 5;
    figcost.gridsize[1] = 5;
    figsize.gridsize[0] = 3;
    figsize.gridsize[1] = 5;
    figtimer.gridsize[0] = 3;
    figtimer.gridsize[1] = 5;

    // minimum ranges
    figconstraint.range[0][0] = 0;
    figconstraint.range[0][1] = 20;
    figconstraint.range[1][0] = 0;
    figconstraint.range[1][1] = 80;
    figcost.range[0][0] = 0;
    figcost.range[0][1] = 20;
    figcost.range[1][0] = -15;
    figcost.range[1][1] = 5;
    figsize.range[0][0] = -200;
    figsize.range[0][1] = 0;
    figsize.range[1][0] = 0;
    figsize.range[1][1] = 100;
    figtimer.range[0][0] = -200;
    figtimer.range[0][1] = 0;
    figtimer.range[1][0] = 0;
    figtimer.range[1][1] = 0.4f;

    // init x axis on history figures (do not show yet)
    for( n=0; n<6; n++ )
        for( i=0; i<mjMAXLINEPNT; i++ )
        {
            figtimer.linedata[n][2*i] = (float)-i;
            figsize.linedata[n][2*i] = (float)-i;
        }
}



// update profiler figures
void profilerupdate(void)
{
    int i, n;

    // update constraint figure
    figconstraint.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
    for( i=1; i<5; i++ )
        figconstraint.linepnt[i] = figconstraint.linepnt[0];
    if( m->opt.solver==mjSOL_PGS )
    {
        figconstraint.linepnt[3] = 0;
        figconstraint.linepnt[4] = 0;
    }
    if( m->opt.solver==mjSOL_CG )
        figconstraint.linepnt[4] = 0;
    for( i=0; i<figconstraint.linepnt[0]; i++ )
    {
        // x
        figconstraint.linedata[0][2*i] = (float)i;
        figconstraint.linedata[1][2*i] = (float)i;
        figconstraint.linedata[2][2*i] = (float)i;
        figconstraint.linedata[3][2*i] = (float)i;
        figconstraint.linedata[4][2*i] = (float)i;

        // y
        figconstraint.linedata[0][2*i+1] = (float)d->nefc;
        figconstraint.linedata[1][2*i+1] = (float)d->solver[i].nactive;
        figconstraint.linedata[2][2*i+1] = (float)d->solver[i].nchange;
        figconstraint.linedata[3][2*i+1] = (float)d->solver[i].neval;
        figconstraint.linedata[4][2*i+1] = (float)d->solver[i].nupdate;
    }

    // update cost figure
    figcost.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
    for( i=1; i<3; i++ )
        figcost.linepnt[i] = figcost.linepnt[0];
    if( m->opt.solver==mjSOL_PGS )
    {
        figcost.linepnt[1] = 0;
        figcost.linepnt[2] = 0;
    }

    for( i=0; i<figcost.linepnt[0]; i++ )
    {
        // x
        figcost.linedata[0][2*i] = (float)i;
        figcost.linedata[1][2*i] = (float)i;
        figcost.linedata[2][2*i] = (float)i;

        // y
        figcost.linedata[0][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].improvement));
        figcost.linedata[1][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].gradient));
        figcost.linedata[2][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].lineslope));
    }

    // get timers: total, collision, prepare, solve, other
    mjtNum total = d->timer[mjTIMER_STEP].duration;
    int number = d->timer[mjTIMER_STEP].number;
    if( !number )
    {
        total = d->timer[mjTIMER_FORWARD].duration;
        number = d->timer[mjTIMER_FORWARD].number;
    }
    number = mjMAX(1, number);
    float tdata[5] = {
        (float)(total/number),
        (float)(d->timer[mjTIMER_POS_COLLISION].duration/number),
        (float)(d->timer[mjTIMER_POS_MAKE].duration/number) +
            (float)(d->timer[mjTIMER_POS_PROJECT].duration/number),
        (float)(d->timer[mjTIMER_CONSTRAINT].duration/number),
        0
    };
    tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

    // update figtimer
    int pnt = mjMIN(201, figtimer.linepnt[0]+1);
    for( n=0; n<5; n++ )
    {
        // shift data
        for( i=pnt-1; i>0; i-- )
            figtimer.linedata[n][2*i+1] = figtimer.linedata[n][2*i-1];

        // assign new
        figtimer.linepnt[n] = pnt;
        figtimer.linedata[n][1] = tdata[n];
    }

    // get sizes: nv, nbody, nefc, sqrt(nnz), ncont, iter
    float sdata[6] = {
        (float)m->nv,
        (float)m->nbody,
        (float)d->nefc,
        (float)mju_sqrt((mjtNum)d->solver_nnz),
        (float)d->ncon,
        (float)d->solver_iter
    };

    // update figsize
    pnt = mjMIN(201, figsize.linepnt[0]+1);
    for( n=0; n<6; n++ )
    {
        // shift data
        for( i=pnt-1; i>0; i-- )
            figsize.linedata[n][2*i+1] = figsize.linedata[n][2*i-1];

        // assign new
        figsize.linepnt[n] = pnt;
        figsize.linedata[n][1] = sdata[n];
    }
}



// show profiler figures
void profilershow(mjrRect rect)
{
    mjrRect viewport = {
        rect.left + rect.width - rect.width/4,
        rect.bottom,
        rect.width/4,
        rect.height/4
    };
    mjr_figure(viewport, &figtimer, &con);
    viewport.bottom += rect.height/4;
    mjr_figure(viewport, &figsize, &con);
    viewport.bottom += rect.height/4;
    mjr_figure(viewport, &figcost, &con);
    viewport.bottom += rect.height/4;
    mjr_figure(viewport, &figconstraint, &con);
}



// init sensor figure
void sensorinit(void)
{
    // set figure to default
    mjv_defaultFigure(&figsensor);
    figsensor.figurergba[3] = 0.5f;

    // set flags
    figsensor.flg_extend = 1;
    figsensor.flg_barplot = 1;
    figsensor.flg_symmetric = 1;

    // title
    strcpy(figsensor.title, "Sensor data");

    // y-tick nubmer format
    strcpy(figsensor.yformat, "%.0f");

    // grid size
    figsensor.gridsize[0] = 2;
    figsensor.gridsize[1] = 3;

    // minimum range
    figsensor.range[0][0] = 0;
    figsensor.range[0][1] = 0;
    figsensor.range[1][0] = -1;
    figsensor.range[1][1] = 1;
}



// update sensor figure
void sensorupdate(void)
{
    static const int maxline = 10;

    // clear linepnt
    for( int i=0; i<maxline; i++ )
        figsensor.linepnt[i] = 0;

    // start with line 0
    int lineid = 0;

    // loop over sensors
    for( int n=0; n<m->nsensor; n++ )
    {
        // go to next line if type is different
        if( n>0 && m->sensor_type[n]!=m->sensor_type[n-1] )
            lineid = mjMIN(lineid+1, maxline-1);

        // get info about this sensor
        mjtNum cutoff = (m->sensor_cutoff[n]>0 ? m->sensor_cutoff[n] : 1);
        int adr = m->sensor_adr[n];
        int dim = m->sensor_dim[n];

        // data pointer in line
        int p = figsensor.linepnt[lineid];

        // fill in data for this sensor
        for( int i=0; i<dim; i++ )
        {
            // check size
            if( (p+2*i)>=mjMAXLINEPNT/2 )
                break;

            // x
            figsensor.linedata[lineid][2*p+4*i] = (float)(adr+i);
            figsensor.linedata[lineid][2*p+4*i+2] = (float)(adr+i);

            // y
            figsensor.linedata[lineid][2*p+4*i+1] = 0;
            figsensor.linedata[lineid][2*p+4*i+3] = (float)(d->sensordata[adr+i]/cutoff);
        }

        // update linepnt
        figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT-1,
                                          figsensor.linepnt[lineid]+2*dim);
    }
}



// show sensor figure
void sensorshow(mjrRect rect)
{
    // constant width with and without profiler
    int width = settings.profiler ? rect.width/3 : rect.width/4;

    // render figure on the right
    mjrRect viewport = {
        rect.left + rect.width - width,
        rect.bottom,
        width,
        rect.height/3
    };
    mjr_figure(viewport, &figsensor, &con);
}



// prepare info text
void infotext(char* title, char* content, double interval)
{
    char tmp[20];

    // compute solver error
    mjtNum solerr = 0;
    if( d->solver_iter )
    {
        int ind = mjMIN(d->solver_iter-1,mjNSOLVER-1);
        solerr = mju_min(d->solver[ind].improvement, d->solver[ind].gradient);
        if( solerr==0 )
            solerr = mju_max(d->solver[ind].improvement, d->solver[ind].gradient);
    }
    solerr = mju_log10(mju_max(mjMINVAL, solerr));

    // prepare info text
    strcpy(title, "Time\nSize\nCPU\nSolver   \nFPS\nstack\nconbuf\nefcbuf");
    sprintf(content, "%-20.3f\n%d  (%d con)\n%.3f\n%.1f  (%d it)\n%.0f\n%.3f\n%.3f\n%.3f",
            d->time,
            d->nefc, d->ncon,
            settings.run ?
                d->timer[mjTIMER_STEP].duration / mjMAX(1, d->timer[mjTIMER_STEP].number) :
                d->timer[mjTIMER_FORWARD].duration / mjMAX(1, d->timer[mjTIMER_FORWARD].number),
            solerr, d->solver_iter,
            1/interval,
            d->maxuse_stack/(double)d->nstack,
            d->maxuse_con/(double)m->nconmax,
            d->maxuse_efc/(double)m->njmax);

    // add Energy if enabled
    if( mjENABLED(mjENBL_ENERGY) )
    {
        sprintf(tmp, "\n%.3f", d->energy[0]+d->energy[1]);
        strcat(content, tmp);
        strcat(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if( mjENABLED(mjENBL_FWDINV) )
    {
        sprintf(tmp, "\n%.1f %.1f",
            mju_log10(mju_max(mjMINVAL,d->solver_fwdinv[0])),
            mju_log10(mju_max(mjMINVAL,d->solver_fwdinv[1])));
        strcat(content, tmp);
        strcat(title, "\nFwdInv");
    }
}



// sprintf forwarding, to avoid compiler warning in x-macro
void printfield(char* str, void* ptr)
{
    sprintf(str, "%g", *(mjtNum*)ptr);
}



// update watch
void watch(void)
{
    // clear
    ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
    strcpy(ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

    // prepare constants for NC
    int nv = m->nv;
    int njmax = m->njmax;

    // find specified field in mjData arrays, update value
    #define X(TYPE, NAME, NR, NC)                                           \
        if( !strcmp(#NAME, settings.field) && !strcmp(#TYPE, "mjtNum") )    \
        {                                                                   \
            if( settings.index>=0 && settings.index<m->NR*NC )              \
                printfield(ui0.sect[SECT_WATCH].item[2].multi.name[0],      \
                           d->NAME + settings.index);                       \
            else                                                            \
                strcpy(ui0.sect[SECT_WATCH].item[2].multi.name[0],          \
                       "invalid index");                                    \
            return;                                                         \
        }

        MJDATA_POINTERS
    #undef X
}



//-------------------------------- UI construction --------------------------------------

// make physics section of UI
void makephysics(int oldstate)
{
    int i;

    mjuiDef defPhysics[] =
    {
        {mjITEM_SECTION,   "Physics",       oldstate, NULL,                 "AP"},
        {mjITEM_SELECT,    "Integrator",    2, &(m->opt.integrator),        "Euler\nRK4"},
        {mjITEM_SELECT,    "Collision",     2, &(m->opt.collision),         "All\nPair\nDynamic"},
        {mjITEM_SELECT,    "Cone",          2, &(m->opt.cone),              "Pyramidal\nElliptic"},
        {mjITEM_SELECT,    "Jacobian",      2, &(m->opt.jacobian),          "Dense\nSparse\nAuto"},
        {mjITEM_SELECT,    "Solver",        2, &(m->opt.solver),            "PGS\nCG\nNewton"},
        {mjITEM_SEPARATOR, "Algorithmic Parameters", 1},
        {mjITEM_EDITNUM,   "Timestep",      2, &(m->opt.timestep),          "1 0 1"},
        {mjITEM_EDITINT,   "Iterations",    2, &(m->opt.iterations),        "1 0 1000"},
        {mjITEM_EDITNUM,   "Tolerance",     2, &(m->opt.tolerance),         "1 0 1"},
        {mjITEM_EDITINT,   "Noslip Iter",   2, &(m->opt.noslip_iterations), "1 0 1000"},
        {mjITEM_EDITNUM,   "Noslip Tol",    2, &(m->opt.noslip_tolerance),  "1 0 1"},
        {mjITEM_EDITINT,   "MRR Iter",      2, &(m->opt.mpr_iterations),    "1 0 1000"},
        {mjITEM_EDITNUM,   "MPR Tol",       2, &(m->opt.mpr_tolerance),     "1 0 1"},
        {mjITEM_EDITNUM,   "API Rate",      2, &(m->opt.apirate),           "1 0 1000"},
        {mjITEM_SEPARATOR, "Physical Parameters", 1},
        {mjITEM_EDITNUM,   "Gravity",       2, m->opt.gravity,              "3"},
        {mjITEM_EDITNUM,   "Wind",          2, m->opt.wind,                 "3"},
        {mjITEM_EDITNUM,   "Magnetic",      2, m->opt.magnetic,             "3"},
        {mjITEM_EDITNUM,   "Density",       2, &(m->opt.density),           "1"},
        {mjITEM_EDITNUM,   "Viscosity",     2, &(m->opt.viscosity),         "1"},
        {mjITEM_EDITNUM,   "Imp Ratio",     2, &(m->opt.impratio),          "1"},
        {mjITEM_SEPARATOR, "Disable Flags", 1},
        {mjITEM_END}
    };
    mjuiDef defEnableFlags[] =
    {
        {mjITEM_SEPARATOR, "Enable Flags", 1},
        {mjITEM_END}
    };
    mjuiDef defOverride[] =
    {
        {mjITEM_SEPARATOR, "Contact Override", 1},
        {mjITEM_EDITNUM,   "Margin",        2, &(m->opt.o_margin),          "1"},
        {mjITEM_EDITNUM,   "Sol Imp",       2, &(m->opt.o_solimp),          "5"},
        {mjITEM_EDITNUM,   "Sol Ref",       2, &(m->opt.o_solref),          "2"},
        {mjITEM_END}
    };

    // add physics
    mjui_add(&ui0, defPhysics);

    // add flags programmatically
    mjuiDef defFlag[] =
    {
        {mjITEM_CHECKINT,  "", 2, NULL, ""},
        {mjITEM_END}
    };
    for( i=0; i<mjNDISABLE; i++ )
    {
        strcpy(defFlag[0].name, mjDISABLESTRING[i]);
        defFlag[0].pdata = settings.disable + i;
        mjui_add(&ui0, defFlag);
    }
    mjui_add(&ui0, defEnableFlags);
    for( i=0; i<mjNENABLE; i++ )
    {
        strcpy(defFlag[0].name, mjENABLESTRING[i]);
        defFlag[0].pdata = settings.enable + i;
        mjui_add(&ui0, defFlag);
    }

    // add contact override
    mjui_add(&ui0, defOverride);
}



// make rendering section of UI
void makerendering(int oldstate)
{
    int i, j;

    mjuiDef defRendering[] =
    {
        {mjITEM_SECTION,    "Rendering",        oldstate, NULL,             "AR"},
        {mjITEM_SELECT,     "Camera",           2, &(settings.camera),      "Free\nTracking"},
        {mjITEM_SELECT,     "Label",            2, &(vopt.label),
            "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\nActuator\nConstraint\nSkin\nSelection\nSel Pnt\nForce"},
        {mjITEM_SELECT,     "Frame",            2, &(vopt.frame),
            "None\nBody\nGeom\nSite\nCamera\nLight\nWorld"},
        {mjITEM_SEPARATOR,  "Model Elements",   1},
        {mjITEM_END}
    };
    mjuiDef defOpenGL[] =
    {
        {mjITEM_SEPARATOR, "OpenGL Effects", 1},
        {mjITEM_END}
    };

    // add model cameras, up to UI limit
    for( i=0; i<mjMIN(m->ncam, mjMAXUIMULTI-2); i++ )
    {
        // prepare name
        char camname[mjMAXUITEXT] = "\n";
        if( m->names[m->name_camadr[i]] )
            strcat(camname, m->names+m->name_camadr[i]);
        else
            sprintf(camname, "\nCamera %d", i);

        // check string length
        if( strlen(camname) + strlen(defRendering[1].other)>=mjMAXUITEXT-1 )
            break;

        // add camera
        strcat(defRendering[1].other, camname);
    }

    // add rendering standard
    mjui_add(&ui0, defRendering);

    // add flags programmatically
    mjuiDef defFlag[] =
    {
        {mjITEM_CHECKBYTE,  "", 2, NULL, ""},
        {mjITEM_END}
    };
    for( i=0; i<mjNVISFLAG; i++ )
    {
        // set name, remove "&"
        strcpy(defFlag[0].name, mjVISSTRING[i][0]);
        for( j=0; j<strlen(mjVISSTRING[i][0]); j++ )
            if( mjVISSTRING[i][0][j]=='&' )
            {
                strcpy(defFlag[0].name+j, mjVISSTRING[i][0]+j+1);
                break;
            }

        // set shortcut and data
        sprintf(defFlag[0].other, " %s", mjVISSTRING[i][2]);
        defFlag[0].pdata = vopt.flags + i;
        mjui_add(&ui0, defFlag);
    }
    mjui_add(&ui0, defOpenGL);
    for( i=0; i<mjNRNDFLAG; i++ )
    {
        strcpy(defFlag[0].name, mjRNDSTRING[i][0]);
        sprintf(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
        defFlag[0].pdata = scn.flags + i;
        mjui_add(&ui0, defFlag);
    }
}



// make group section of UI
void makegroup(int oldstate)
{
    mjuiDef defGroup[] =
    {
        {mjITEM_SECTION,    "Group enable",     oldstate, NULL,             "AG"},
        {mjITEM_SEPARATOR,  "Geom groups",  1},
        {mjITEM_CHECKBYTE,  "Geom 0",           2, vopt.geomgroup,          " 0"},
        {mjITEM_CHECKBYTE,  "Geom 1",           2, vopt.geomgroup+1,        " 1"},
        {mjITEM_CHECKBYTE,  "Geom 2",           2, vopt.geomgroup+2,        " 2"},
        {mjITEM_CHECKBYTE,  "Geom 3",           2, vopt.geomgroup+3,        " 3"},
        {mjITEM_CHECKBYTE,  "Geom 4",           2, vopt.geomgroup+4,        " 4"},
        {mjITEM_CHECKBYTE,  "Geom 5",           2, vopt.geomgroup+5,        " 5"},
        {mjITEM_SEPARATOR,  "Site groups",  1},
        {mjITEM_CHECKBYTE,  "Site 0",           2, vopt.sitegroup,          "S0"},
        {mjITEM_CHECKBYTE,  "Site 1",           2, vopt.sitegroup+1,        "S1"},
        {mjITEM_CHECKBYTE,  "Site 2",           2, vopt.sitegroup+2,        "S2"},
        {mjITEM_CHECKBYTE,  "Site 3",           2, vopt.sitegroup+3,        "S3"},
        {mjITEM_CHECKBYTE,  "Site 4",           2, vopt.sitegroup+4,        "S4"},
        {mjITEM_CHECKBYTE,  "Site 5",           2, vopt.sitegroup+5,        "S5"},
        {mjITEM_SEPARATOR,  "Joint groups", 1},
        {mjITEM_CHECKBYTE,  "Joint 0",          2, vopt.jointgroup,         ""},
        {mjITEM_CHECKBYTE,  "Joint 1",          2, vopt.jointgroup+1,       ""},
        {mjITEM_CHECKBYTE,  "Joint 2",          2, vopt.jointgroup+2,       ""},
        {mjITEM_CHECKBYTE,  "Joint 3",          2, vopt.jointgroup+3,       ""},
        {mjITEM_CHECKBYTE,  "Joint 4",          2, vopt.jointgroup+4,       ""},
        {mjITEM_CHECKBYTE,  "Joint 5",          2, vopt.jointgroup+5,       ""},
        {mjITEM_SEPARATOR,  "Tendon groups",    1},
        {mjITEM_CHECKBYTE,  "Tendon 0",         2, vopt.tendongroup,        ""},
        {mjITEM_CHECKBYTE,  "Tendon 1",         2, vopt.tendongroup+1,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 2",         2, vopt.tendongroup+2,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 3",         2, vopt.tendongroup+3,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 4",         2, vopt.tendongroup+4,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 5",         2, vopt.tendongroup+5,      ""},
        {mjITEM_SEPARATOR,  "Actuator groups", 1},
        {mjITEM_CHECKBYTE,  "Actuator 0",       2, vopt.actuatorgroup,      ""},
        {mjITEM_CHECKBYTE,  "Actuator 1",       2, vopt.actuatorgroup+1,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 2",       2, vopt.actuatorgroup+2,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 3",       2, vopt.actuatorgroup+3,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 4",       2, vopt.actuatorgroup+4,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 5",       2, vopt.actuatorgroup+5,    ""},
        {mjITEM_END}
    };

    // add section
    mjui_add(&ui0, defGroup);
}



// make joint section of UI
void makejoint(int oldstate)
{
    int i;

    mjuiDef defJoint[] =
    {
        {mjITEM_SECTION, "Joint", oldstate, NULL, "AJ"},
        {mjITEM_END}
    };
    mjuiDef defSlider[] =
    {
        {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
        {mjITEM_END}
    };

    // add section
    mjui_add(&ui1, defJoint);
    defSlider[0].state = 4;

    // add scalar joints, exit if UI limit reached
    int itemcnt = 0;
    for( i=0; i<m->njnt && itemcnt<mjMAXUIITEM; i++ )
        if( (m->jnt_type[i]==mjJNT_HINGE || m->jnt_type[i]==mjJNT_SLIDE) )
        {
            // skip if joint group is disabled
            if( !vopt.jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->jnt_group[i]))] )
                continue;

            // set data and name
            defSlider[0].pdata = d->qpos + m->jnt_qposadr[i];
            if( m->names[m->name_jntadr[i]] )
                mju_strncpy(defSlider[0].name, m->names+m->name_jntadr[i],
                            mjMAXUINAME);
            else
                sprintf(defSlider[0].name, "joint %d", i);

            // set range
            if( m->jnt_limited[i] )
                sprintf(defSlider[0].other, "%.4g %.4g",
                    m->jnt_range[2*i], m->jnt_range[2*i+1]);
            else if( m->jnt_type[i]==mjJNT_SLIDE )
                strcpy(defSlider[0].other, "-1 1");
            else
                strcpy(defSlider[0].other, "-3.1416 3.1416");

            // add and count
            mjui_add(&ui1, defSlider);
            itemcnt++;
        }
}



// make control section of UI
void makecontrol(int oldstate)
{
    int i;

    mjuiDef defControl[] =
    {
        {mjITEM_SECTION, "Control", oldstate, NULL, "AC"},
        {mjITEM_BUTTON,  "Clear all", 2},
        {mjITEM_END}
    };
    mjuiDef defSlider[] =
    {
        {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
        {mjITEM_END}
    };

    // add section
    mjui_add(&ui1, defControl);
    defSlider[0].state = 2;

    // add controls, exit if UI limit reached (Clear button already added)
    int itemcnt = 1;
    for( i=0; i<m->nu && itemcnt<mjMAXUIITEM; i++ )
    {
        // skip if actuator group is disabled
        if( !vopt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->actuator_group[i]))] )
            continue;

        // set data and name
        defSlider[0].pdata = d->ctrl + i;
        if( m->names[m->name_actuatoradr[i]] )
            mju_strncpy(defSlider[0].name, m->names+m->name_actuatoradr[i],
                        mjMAXUINAME);
        else
            sprintf(defSlider[0].name, "control %d", i);

        // set range
        if( m->actuator_ctrllimited[i] )
            sprintf(defSlider[0].other, "%.4g %.4g",
                m->actuator_ctrlrange[2*i], m->actuator_ctrlrange[2*i+1]);
        else
            strcpy(defSlider[0].other, "-1 1");

        // add and count
        mjui_add(&ui1, defSlider);
        itemcnt++;
    }
}



// make model-dependent UI sections
void makesections(void)
{
    int i;

    // get section open-close state, UI 0
    int oldstate0[NSECT0];
    for( i=0; i<NSECT0; i++ )
    {
        oldstate0[i] = 0;
        if( ui0.nsect>i )
            oldstate0[i] = ui0.sect[i].state;
    }

    // get section open-close state, UI 1
    int oldstate1[NSECT1];
    for( i=0; i<NSECT1; i++ )
    {
        oldstate1[i] = 0;
        if( ui1.nsect>i )
            oldstate1[i] = ui1.sect[i].state;
    }

    // clear model-dependent sections of UI
    ui0.nsect = SECT_PHYSICS;
    ui1.nsect = 0;

    // make
    makephysics(oldstate0[SECT_PHYSICS]);
    makerendering(oldstate0[SECT_RENDERING]);
    makegroup(oldstate0[SECT_GROUP]);
    makejoint(oldstate1[SECT_JOINT]);
    makecontrol(oldstate1[SECT_CONTROL]);
}



//-------------------------------- utility functions ------------------------------------

// align and scale view
void alignscale(void)
{
    // autoscale
    cam.lookat[0] = m->stat.center[0];
    cam.lookat[1] = m->stat.center[1];
    cam.lookat[2] = m->stat.center[2];
    cam.distance = 1.5 * m->stat.extent;

    // set to free camera
    cam.type = mjCAMERA_FREE;
}



// copy qpos to clipboard as key
void copykey(void)
{
    char clipboard[5000] = "<key qpos='";
    char buf[200];

    // prepare string
    for( int i=0; i<m->nq; i++ )
    {
        sprintf(buf, i==m->nq-1 ? "%g" : "%g ", d->qpos[i]);
        strcat(clipboard, buf);
    }
    strcat(clipboard, "'/>");

    // copy to clipboard
    glfwSetClipboardString(window, clipboard);
}



// millisecond timer, for MuJoCo built-in profiler
mjtNum timer(void)
{
    return (mjtNum)(1000*glfwGetTime());
}



// clear all times
void cleartimers(void)
{
    for( int i=0; i<mjNTIMER; i++ )
    {
        d->timer[i].duration = 0;
        d->timer[i].number = 0;
    }
}



// update UI 0 when MuJoCo structures change (except for joint sliders)
void updatesettings(void)
{
    int i;

    // physics flags
    for( i=0; i<mjNDISABLE; i++ )
        settings.disable[i] = ((m->opt.disableflags & (1<<i)) !=0 );
    for( i=0; i<mjNENABLE; i++ )
        settings.enable[i] = ((m->opt.enableflags & (1<<i)) !=0 );

    // camera
    if( cam.type==mjCAMERA_FIXED )
        settings.camera = 2 + cam.fixedcamid;
    else if( cam.type==mjCAMERA_TRACKING )
        settings.camera = 1;
    else
        settings.camera = 0;

    // update UI
    mjui_update(-1, -1, &ui0, &uistate, &con);
}



// drop file callback
void drop(GLFWwindow* window, int count, const char** paths)
{
    // make sure list is non-empty
    if( count>0 )
    {
        mju_strncpy(filename, paths[0], 1000);
        settings.loadrequest = 1;
    }
}



// load mjb or xml model
void loadmodel(void)
{
    // clear request
    settings.loadrequest = 0;

    // make sure filename is not empty
    if( !filename[0]  )
        return;

    // load and compile
    char error[500] = "";
    mjModel* mnew = 0;
    if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
    {
        mnew = mj_loadModel(filename, NULL);
        if( !mnew )
            strcpy(error, "could not load binary model");
    }
    else
        mnew = mj_loadXML(filename, NULL, error, 500);
    if( !mnew )
    {
        printf("%s\n", error);
        return;
    }

    // compiler warning: print and pause
    if( error[0] )
    {
        // mj_forward() below will print the warning message
        printf("Model compiled, but simulation warning (paused):\n  %s\n\n",
                error);
        settings.run = 0;
    }

    // delete old model, assign new
    mj_deleteData(d);
    mj_deleteModel(m);
    m = mnew;
    d = mj_makeData(m);
    mj_forward(m, d);

    // re-create scene and context
    mjv_makeScene(m, &scn, maxgeom);
    mjr_makeContext(m, &con, 50*(settings.font+1));

    // clear perturbation state
    pert.active = 0;
    pert.select = 0;
    pert.skinselect = -1;

    // align and scale view, update scene
    alignscale();
    mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

    // set window title to model name
    if( window && m->names )
    {
        char title[200] = "Simulate : ";
        strcat(title, m->names);
        glfwSetWindowTitle(window, title);
    }

    // set keyframe range and divisions
    ui0.sect[SECT_SIMULATION].item[6].slider.range[0] = 0;
    ui0.sect[SECT_SIMULATION].item[6].slider.range[1] = mjMAX(0, m->nkey - 1);
    ui0.sect[SECT_SIMULATION].item[6].slider.divisions = mjMAX(1, m->nkey - 1);

    // rebuild UI sections
    makesections();

    // full ui update
    uiModify(window, &ui0, &uistate, &con);
    uiModify(window, &ui1, &uistate, &con);
    updatesettings();
}



//--------------------------------- UI hooks (for uitools.c) ----------------------------

// determine enable/disable item state given category
int uiPredicate(int category, void* userdata)
{
    switch( category )
    {
    case 2:                 // require model
        return (m!=NULL);

    case 3:                 // require model and nkey
        return (m && m->nkey);

    case 4:                 // require model and paused
        return (m && !settings.run);

    default:
        return 1;
    }
}



// set window layout
void uiLayout(mjuiState* state)
{
    mjrRect* rect = state->rect;

    // set number of rectangles
    state->nrect = 4;

    // rect 0: entire framebuffer
    rect[0].left = 0;
    rect[0].bottom = 0;
    glfwGetFramebufferSize(window, &rect[0].width, &rect[0].height);

    // rect 1: UI 0
    rect[1].left = 0;
    rect[1].width = settings.ui0 ? ui0.width : 0;
    rect[1].bottom = 0;
    rect[1].height = rect[0].height;

    // rect 2: UI 1
    rect[2].width = settings.ui1 ? ui1.width : 0;
    rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
    rect[2].bottom = 0;
    rect[2].height = rect[0].height;

    // rect 3: 3D plot (everything else is an overlay)
    rect[3].left = rect[1].width;
    rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
    rect[3].bottom = 0;
    rect[3].height = rect[0].height;
}



// handle UI event
void uiEvent(mjuiState* state)
{
    int i;
    char err[200];

    // call UI 0 if event is directed to it
    if( (state->dragrect==ui0.rectid) ||
        (state->dragrect==0 && state->mouserect==ui0.rectid) ||
        state->type==mjEVENT_KEY )
    {
        // process UI event
        mjuiItem* it = mjui_event(&ui0, state, &con);

        // file section
        if( it && it->sectionid==SECT_FILE )
        {
            switch( it->itemid )
            {
            case 0:             // Save xml
                if( !mj_saveLastXML("mjmodel.xml", m, err, 200) )
                    printf("Save XML error: %s", err);
                break;

            case 1:             // Save mjb
                mj_saveModel(m, "mjmodel.mjb", NULL, 0);
                break;

            case 2:             // Print model
                mj_printModel(m, "MJMODEL.TXT");
                break;

            case 3:             // Print data
                mj_printData(m, d, "MJDATA.TXT");
                break;

            case 4:             // Quit
                settings.exitrequest = 1;
                break;
            }
        }

        // option section
        else if( it && it->sectionid==SECT_OPTION )
        {
            switch( it->itemid )
            {
            case 0:             // Spacing
                ui0.spacing = mjui_themeSpacing(settings.spacing);
                ui1.spacing = mjui_themeSpacing(settings.spacing);
                break;

            case 1:             // Color
                ui0.color = mjui_themeColor(settings.color);
                ui1.color = mjui_themeColor(settings.color);
                break;

            case 2:             // Font
                mjr_changeFont(50*(settings.font+1), &con);
                break;

            case 9:             // Full screen
                if( glfwGetWindowMonitor(window) )
                {
                    // restore window from saved data
                    glfwSetWindowMonitor(window, NULL, windowpos[0], windowpos[1],
                                         windowsize[0], windowsize[1], 0);
                }

                // currently windowed: switch to full screen
                else
                {
                    // save window data
                    glfwGetWindowPos(window, windowpos, windowpos+1);
                    glfwGetWindowSize(window, windowsize, windowsize+1);

                    // switch
                    glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0,
                                         vmode.width, vmode.height, vmode.refreshRate);
                }

                // reinstante vsync, just in case
                glfwSwapInterval(settings.vsync);
                break;

            case 10:            // Vertical sync
                glfwSwapInterval(settings.vsync);
                break;
            }

            // modify UI
            uiModify(window, &ui0, state, &con);
            uiModify(window, &ui1, state, &con);
        }

        // simulation section
        else if( it && it->sectionid==SECT_SIMULATION )
        {
            switch( it->itemid )
            {
            case 1:             // Reset
                if( m )
                {
                    mj_resetData(m, d);
                    mj_forward(m, d);
                    profilerupdate();
                    sensorupdate();
                    updatesettings();
                }
                break;

            case 2:             // Reload
                settings.loadrequest = 1;
                break;

            case 3:             // Align
                alignscale();
                updatesettings();
                break;

            case 4:             // Copy pose
                copykey();
                break;

            case 5:             // Adjust key
            case 6:             // Reset to key
                i = settings.key;
                d->time = m->key_time[i];
                mju_copy(d->qpos, m->key_qpos+i*m->nq, m->nq);
                mju_copy(d->qvel, m->key_qvel+i*m->nv, m->nv);
                mju_copy(d->act, m->key_act+i*m->na, m->na);
                mj_forward(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();
                break;

            case 7:             // Set key
                i = settings.key;
                m->key_time[i] = d->time;
                mju_copy(m->key_qpos+i*m->nq, d->qpos, m->nq);
                mju_copy(m->key_qvel+i*m->nv, d->qvel, m->nv);
                mju_copy(m->key_act+i*m->na, d->act, m->na);
                break;
            }
        }

        // physics section
        else if( it && it->sectionid==SECT_PHYSICS )
        {
            // update disable flags in mjOption
            m->opt.disableflags = 0;
            for( i=0; i<mjNDISABLE; i++ )
                if( settings.disable[i] )
                    m->opt.disableflags |= (1<<i);

            // update enable flags in mjOption
            m->opt.enableflags = 0;
            for( i=0; i<mjNENABLE; i++ )
                if( settings.enable[i] )
                    m->opt.enableflags |= (1<<i);
        }

        // rendering section
        else if( it && it->sectionid==SECT_RENDERING )
        {
            // set camera in mjvCamera
            if( settings.camera==0 )
                cam.type = mjCAMERA_FREE;
            else if( settings.camera==1 )
            {
                if( pert.select>0 )
                {
                    cam.type = mjCAMERA_TRACKING;
                    cam.trackbodyid = pert.select;
                    cam.fixedcamid = -1;
                }
                else
                {
                    cam.type = mjCAMERA_FREE;
                    settings.camera = 0;
                    mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
                }
            }
            else
            {
                cam.type = mjCAMERA_FIXED;
                cam.fixedcamid = settings.camera - 2;
            }
        }

        // group section
        else if( it && it->sectionid==SECT_GROUP )
        {
            // remake joint section if joint group changed
            if( it->name[0]=='J' && it->name[1]=='o' )
            {
                ui1.nsect = SECT_JOINT;
                makejoint(ui1.sect[SECT_JOINT].state);
                ui1.nsect = NSECT1;
                uiModify(window, &ui1, state, &con);
            }

            // remake control section if actuator group changed
            if( it->name[0]=='A' && it->name[1]=='c' )
            {
                ui1.nsect = SECT_CONTROL;
                makecontrol(ui1.sect[SECT_CONTROL].state);
                ui1.nsect = NSECT1;
                uiModify(window, &ui1, state, &con);
            }
        }

        // stop if UI processed event
        if( it!=NULL || (state->type==mjEVENT_KEY && state->key==0) )
            return;
    }

    // call UI 1 if event is directed to it
    if( (state->dragrect==ui1.rectid) ||
        (state->dragrect==0 && state->mouserect==ui1.rectid) ||
        state->type==mjEVENT_KEY )
    {
        // process UI event
        mjuiItem* it = mjui_event(&ui1, state, &con);

        // control section
        if( it && it->sectionid==SECT_CONTROL )
        {
            // clear controls
            if( it->itemid==0 )
            {
                mju_zero(d->ctrl, m->nu);
                mjui_update(SECT_CONTROL, -1, &ui1, &uistate, &con);
            }
        }

        // stop if UI processed event
        if( it!=NULL || (state->type==mjEVENT_KEY && state->key==0) )
            return;
    }

    // shortcut not handled by UI
    if( state->type==mjEVENT_KEY && state->key!=0 )
    {
        switch( state->key )
        {
        case ' ':                   // Mode
            if( m )
            {
                settings.run = 1 - settings.run;
                pert.active = 0;
                mjui_update(-1, -1, &ui0, state, &con);
            }
            break;

        case mjKEY_RIGHT:           // step forward
            if( m && !settings.run )
            {
                cleartimers();
                mj_step(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();
            }
            break;

        case mjKEY_LEFT:            // step back
            if( m && !settings.run )
            {
                m->opt.timestep = -m->opt.timestep;
                cleartimers();
                mj_step(m, d);
                m->opt.timestep = -m->opt.timestep;
                profilerupdate();
                sensorupdate();
                updatesettings();
            }
            break;

        case mjKEY_DOWN:            // step forward 100
            if( m && !settings.run )
            {
                cleartimers();
                for( i=0; i<100; i++ )
                    mj_step(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();
            }
            break;

        case mjKEY_UP:              // step back 100
            if( m && !settings.run )
            {
                m->opt.timestep = -m->opt.timestep;
                cleartimers();
                for( i=0; i<100; i++ )
                    mj_step(m, d);
                m->opt.timestep = -m->opt.timestep;
                profilerupdate();
                sensorupdate();
                updatesettings();
            }
            break;

        case mjKEY_PAGE_UP:         // select parent body
            if( m && pert.select>0 )
            {
                pert.select = m->body_parentid[pert.select];
                pert.skinselect = -1;

                // stop perturbation if world reached
                if( pert.select<=0 )
                    pert.active = 0;
            }

            break;

        case mjKEY_ESCAPE:          // free camera
            cam.type = mjCAMERA_FREE;
            settings.camera = 0;
            mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
            break;
        }

        return;
    }

    // 3D scroll
    if( state->type==mjEVENT_SCROLL && state->mouserect==3 && m )
    {
        // emulate vertical mouse motion = 5% of window height
        mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*state->sy, &scn, &cam);

        return;
    }

    // 3D press
    if( state->type==mjEVENT_PRESS && state->mouserect==3 && m )
    {
        // set perturbation
        int newperturb = 0;
        if( state->control && pert.select>0 )
        {
            // right: translate;  left: rotate
            if( state->right )
                newperturb = mjPERT_TRANSLATE;
            else if( state->left )
                newperturb = mjPERT_ROTATE;

            // perturbation onset: reset reference
            if( newperturb && !pert.active )
                mjv_initPerturb(m, d, &scn, &pert);
        }
        pert.active = newperturb;

        // handle double-click
        if( state->doubleclick )
        {
            // determine selection mode
            int selmode;
            if( state->button==mjBUTTON_LEFT )
                selmode = 1;
            else if( state->control )
                selmode = 3;
            else
                selmode = 2;

            // find geom and 3D click point, get corresponding body
            mjrRect r = state->rect[3];
            mjtNum selpnt[3];
            int selgeom, selskin;
            int selbody = mjv_select(m, d, &vopt,
                                     (mjtNum)r.width/(mjtNum)r.height,
                                     (mjtNum)(state->x-r.left)/(mjtNum)r.width,
                                     (mjtNum)(state->y-r.bottom)/(mjtNum)r.height,
                                     &scn, selpnt, &selgeom, &selskin);

            // set lookat point, start tracking is requested
            if( selmode==2 || selmode==3 )
            {
                // copy selpnt if anything clicked
                if( selbody>=0 )
                    mju_copy3(cam.lookat, selpnt);

                // switch to tracking camera if dynamic body clicked
                if( selmode==3 && selbody>0 )
                {
                    // mujoco camera
                    cam.type = mjCAMERA_TRACKING;
                    cam.trackbodyid = selbody;
                    cam.fixedcamid = -1;

                    // UI camera
                    settings.camera = 1;
                    mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
                }
            }

            // set body selection
            else
            {
                if( selbody>=0 )
                {
                    // record selection
                    pert.select = selbody;
                    pert.skinselect = selskin;

                    // compute localpos
                    mjtNum tmp[3];
                    mju_sub3(tmp, selpnt, d->xpos+3*pert.select);
                    mju_mulMatTVec(pert.localpos, d->xmat+9*pert.select, tmp, 3, 3);
                }
                else
                {
                    pert.select = 0;
                    pert.skinselect = -1;
                }
            }

            // stop perturbation on select
            pert.active = 0;
        }

        return;
    }

    // 3D release
    if( state->type==mjEVENT_RELEASE && state->dragrect==3 && m )
    {
        // stop perturbation
        pert.active = 0;

        return;
    }

    // 3D move
    if( state->type==mjEVENT_MOVE && state->dragrect==3 && m )
    {
        // determine action based on mouse button
        mjtMouse action;
        if( state->right )
            action = state->shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        else if( state->left )
            action = state->shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        else
            action = mjMOUSE_ZOOM;

        // move perturb or camera
        mjrRect r = state->rect[3];
        if( pert.active )
            mjv_movePerturb(m, d, action, state->dx/r.height, -state->dy/r.height,
                            &scn, &pert);
        else
            mjv_moveCamera(m, action, state->dx/r.height, -state->dy/r.height,
                           &scn, &cam);

        return;
    }
}



//--------------------------- rendering and simulation ----------------------------------

// sim thread synchronization
std::mutex mtx;


// prepare to render
void prepare(void)
{
    // data for FPS calculation
    static double lastupdatetm = 0;

    // update interval, save update time
    double tmnow = glfwGetTime();
    double interval = tmnow - lastupdatetm;
    interval = mjMIN(1, mjMAX(0.0001, interval));
    lastupdatetm = tmnow;

    // no model: nothing to do
    if( !m )
        return;

    // update scene
    mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

    // update watch
    if( settings.ui0 && ui0.sect[SECT_WATCH].state )
    {
		watch();
		mjui_update(SECT_WATCH, -1, &ui0, &uistate, &con);
    }

    // ipdate joint
    if( settings.ui1 && ui1.sect[SECT_JOINT].state )
            mjui_update(SECT_JOINT, -1, &ui1, &uistate, &con);

    // update info text
    if( settings.info )
        infotext(info_title, info_content, interval);

    // update profiler
    if( settings.profiler && settings.run )
        profilerupdate();

    // update sensor
    if( settings.sensor && settings.run )
        sensorupdate();

    // clear timers once profiler info has been copied
    cleartimers();
}



// render im main thread (while simulating in background thread)
void render(GLFWwindow* window)
{
    // get 3D rectangle and reduced for profiler
    mjrRect rect = uistate.rect[3];
    mjrRect smallrect = rect;
    if( settings.profiler )
        smallrect.width = rect.width - rect.width/4;

    // no model
    if( !m )
    {
        // blank screen
        mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

        // label
        if( settings.loadrequest )
            mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                        "loading", NULL, &con);
        else
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect,
                        "Drag-and-drop model file here", 0, &con);

        // render uis
        if( settings.ui0 )
            mjui_render(&ui0, &uistate, &con);
        if( settings.ui1 )
            mjui_render(&ui1, &uistate, &con);

        // finalize
        glfwSwapBuffers(window);

        return;
    }

    // render scene
    mjr_render(rect, &scn, &con);

    // show pause/loading label
    if( !settings.run || settings.loadrequest )
        mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                    settings.loadrequest ? "loading" : "pause", NULL, &con);

    // show ui 0
    if( settings.ui0 )
        mjui_render(&ui0, &uistate, &con);

    // show ui 1
    if( settings.ui1 )
        mjui_render(&ui1, &uistate, &con);

    // show help
    if( settings.help )
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content, &con);

    // show info
    if( settings.info )
        mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect,
                    info_title, info_content, &con);

    // show profiler
    if( settings.profiler )
        profilershow(rect);

    // show sensor
    if( settings.sensor )
        sensorshow(smallrect);

    // finalize
    glfwSwapBuffers(window);
}



// simulate in background thread (while rendering in main thread)
void simulate(void)
{
    // cpu-sim syncronization point
    double cpusync = 0;
    mjtNum simsync = 0;

    // run until asked to exit
    while( !settings.exitrequest )
    {
        // sleep for 1 ms or yield, to let main thread run
        //  yield results in busy wait - which has better timing but kills battery life
        if( settings.run && settings.busywait )
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // start exclusive access
        mtx.lock();

        // run only if model is present
        if( m )
        {
            // record start time
            double startwalltm = glfwGetTime();

            // running
            if( settings.run )
            {
                // record cpu time at start of iteration
                double tmstart = glfwGetTime();

                // out-of-sync (for any reason)
                if( d->time<simsync || tmstart<cpusync || cpusync==0 ||
                    mju_abs((d->time-simsync)-(tmstart-cpusync))>syncmisalign )
                {
                    // re-sync
                    cpusync = tmstart;
                    simsync = d->time;

                    // clear old perturbations, apply new
                    mju_zero(d->xfrc_applied, 6*m->nbody);
                    mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
                    mjv_applyPerturbForce(m, d, &pert);

                    // run single step, let next iteration deal with timing
                    mj_step(m, d);
                }

                // in-sync
                else
                {
                    // step while simtime lags behind cputime, and within safefactor
                    while( (d->time-simsync)<(glfwGetTime()-cpusync) &&
                           (glfwGetTime()-tmstart)<refreshfactor/vmode.refreshRate )
                    {
                        // clear old perturbations, apply new
                        mju_zero(d->xfrc_applied, 6*m->nbody);
                        mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
                        mjv_applyPerturbForce(m, d, &pert);

                        // run mj_step
                        mjtNum prevtm = d->time;
                        mj_step(m, d);

                        // break on reset
                        if( d->time<prevtm )
                            break;
                    }
                }
            }

            // paused
            else
            {
                // apply pose perturbation
                mjv_applyPerturbPose(m, d, &pert, 1);      // move mocap and dynamic bodies

                // run mj_forward, to update rendering and joint sliders
                mj_forward(m, d);
            }
        }

        // end exclusive access
        mtx.unlock();
    }
}


//---------------------------------------------------------------------------------------
//-------------------------------- AS additions -----------------------------------------
//---------------------------------------------------------------------------------------

//-------------------------------- array and matrix calulcations ------------------------
float arr_max(const float* arr, const int len)
{
    float max=arr[0];
    for (int i = 1; i < len; ++i)
        if (arr[i] > max)
            max = arr[i];
    return max;
}


float arr_max(const float* arr, const int len, const int* indices)
{
    float max=arr[indices[0]];
    int it;
    for (int i = 1; i < len; ++i) {
        it = indices[i];
        if (arr[i] > max)
            max = arr[i];
    }
    return max;
}


float rms(const int len, const float* arr1, const float* arr2)
{
    float val = 0;
    for (int i = 0; i < len; ++i)
        val += pow(arr1[i] - arr2[i], (float)2.0);
    val /= len;
    val = pow(val, 0.5);
    return val;
}


float rms(const int len, const float* arr)
{
    float val = 0;
    for (int i = 0; i < len; ++i)
        val += pow(arr[i], (float)2.0);
    val /= len;
    val = pow(val, 0.5);
    return val;
}


void normalize_arr(float* arr, const int len)
{
    float maxval = -1;
    for (int i = 0; i < len; ++i)
        if (abs(arr[i]) > maxval)
            maxval = abs(arr[i]);

    if (maxval == 0)
        for (int i = 0; i < len; ++i)
            arr[i] = 0;
    else
        for (int i = 0; i < len; ++i)
            arr[i] /= maxval;
}


bool val_in_array(const int val, const int len, const int* arr)
{
    bool answ = false;
    for (int i = 0; i < len; ++i)
        if (arr[i] == val) {
            answ = true;
            break;
        }
    return answ;
}


bool val_in_array(const int val, const std::vector<int> arr)
{
    bool answ = true;
    if (arr.end() == std::find(arr.begin(), arr.end(), val))
        answ = false;
    return answ;
}


float median(std::vector<float> arr)
{
    std::sort(arr.begin(), arr.end());
    return arr[arr.size()/2];
}


float median(std::vector<int> arr)
{
    std::sort(arr.begin(), arr.end());
    return arr[arr.size()/2];
}


bool median(std::vector<bool> arr)
{
    struct {
        bool operator()(bool a, bool b) const
        {
            return (int)a < (int) b;
        }
    } customLess;
    std::sort(arr.begin(), arr.end(), customLess);
    return arr[arr.size()/2];
}


float scalar_multiplication(const int len, const float* arr1, const float* arr2)
{
    float answ=0;
    for (int i = 0; i < len; ++i)
        answ += arr1[i] * arr2[i];
    return answ;
}


//-------------------------------- load and export kinematics ---------------------------
std::vector<std::string> load_mot_file(const char *filename, int *M, int *N, mjtNum** nTime, mjtNum** nPos)
{
    std::ifstream infile(filename);

    std::string line, buf;
    std::getline(infile, line);  // Coordinates
    std::getline(infile, line);  // version

    std::getline(infile, line);
    *N = std::stoi(line.substr(line.find('=')+1));

    std::getline(infile, line);
    *M = std::stoi(line.substr(line.find('=')+1))-1;

    while (line != "endheader")
        std::getline(infile, line);
    std::getline(infile, line);

    std::stringstream ls(line);
    ls >> buf;  // time
    std::vector<std::string> dof_names;
    while (ls >> buf)
        dof_names.push_back(buf);

    *nTime = new mjtNum[(*N)];
    *nPos = new mjtNum[(*N)*(*M)];

    int iTime=0;
    while (std::getline(infile, line)) {
        ls = std::stringstream(line);
        ls >> (*nTime)[iTime];

        for (int i = 0; i < *M; ++i)
            ls >> (*nPos)[iTime*(*M)+i];

        iTime++;
    }

    std::cout << "Read kinematics file" << std::endl;
    std::cout << "\tN = " << *N << std::endl;
    std::cout << "\tM = " << *M << std::endl;

    // std::cout << "\tDOFs: ";
    // for (int i = 0; i < dof_names.size(); ++i)
    //     std::cout << dof_names[i] << '\t';
    // std::cout << std::endl;

    // std::cout << "\tnTime: ";
    // for (int i = 0; i < *N; ++i)
    //     std::cout << (*nTime)[i] << '\t';
    // std::cout << std::endl;

    // std::cout << "\tnPos:" << std::endl;
    // for (int i = 0; i < *N; ++i){
    //     for (int j = 0; j < *M; ++j)
    //         std::cout << (*nPos)[i*(*M)+j] << '\t';
    //     std::cout << std::endl;
    // }

    return dof_names;
}


mjtNum* average_position(const mjtNum* nPos, const int M, const int N)
{
    mjtNum* nPosAverage = new mjtNum[M];
    for (int j = 0; j < M; ++j)
    {
        nPosAverage[j] = 0;
        for (int i = 0; i < N; ++i)
            nPosAverage[j] += nPos[i*M+j];
        nPosAverage[j] /= M;
    }
    return nPosAverage;
}


std::vector<int> map_dofs(const std::vector<std::string> dof_names)
{
    std::vector<int> dof_map(dof_names.size());
    for (int i = 0; i < dof_names.size(); ++i)
        dof_map[i] = -1;

    char buf[mjMAXUINAME];
    std::cout << "Model DOFs: ";
    for (int i = 0; i < m->nq; ++i)
    {
        mju_strncpy(buf, m->names+m->name_jntadr[i], mjMAXUINAME);
        for (int j = 0; j < dof_names.size(); ++j)
            if (dof_names[j] == buf){
                dof_map[j] = i;
                break;
            }
        std::cout << i << ": " << buf << "\t";
    }
    std::cout << std::endl;

    // for (int i = 0; i < dof_names.size(); ++i)
    //     std::cout << dof_map[i] << '\t';
    // std::cout << std::endl;

    return dof_map;
}


void assume_posture(const int M, const std::vector<int> dof_map, const mjtNum* nPosture,
                    const std::vector<mjtNum> dof_multiplier, const std::vector<mjtNum> dof_offset)
{
    for (int i = 0; i < M; ++i)
        if (dof_map[i] >= 0)
            d->qpos[dof_map[i]] = dof_multiplier[dof_map[i]]*nPosture[i] + dof_offset[dof_map[i]];
}


//-------------------------------- contacts ---------------------------------------------
void print_contacts(void)
{
    char buf[mjMAXUINAME];
    float* rgba;
    int bodygeom;
    int tpgeom;

    int ncon = 0;
    for (int i = 0; i < d->ncon; ++i)
    {
        if (d->contact[i].dist > 0)
            continue;

        tpgeom = -1;

        // print the text
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[d->contact[i].geom1]], mjMAXUINAME);
        std::cout << "\t\tGeom1 #" << m->geom_bodyid[d->contact[i].geom1] << " " << buf << " ";
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length())) {
            tpgeom = m->geom_bodyid[d->contact[i].geom1];
            bodygeom = m->geom_bodyid[d->contact[i].geom2];
        }

        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[d->contact[i].geom2]], mjMAXUINAME);
        std::cout << "Geom2 #" << m->geom_bodyid[d->contact[i].geom2] << " " << buf << " ";
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length())) {
            tpgeom = m->geom_bodyid[d->contact[i].geom2];
            bodygeom = m->geom_bodyid[d->contact[i].geom1];
        }

        std::cout << "dist: " << d->contact[i].dist << " ";

        // color them
        if (tpgeom >= 0) {
            rgba = m->geom_rgba + 4*tpgeom;
            std::cout << "rgba: " << rgba[0] << " " << rgba[1] << " " << rgba[2] << " " << rgba[3];
            std::cout << " dim: " << d->contact[i].dim;
            ncon++;
        }

        std::cout << std::endl;
    }
    if (ncon > 0)
        std::cout << "\tContacts detected: " << ncon << std::endl;
}


void print_all_contacts(void)
{
    char buf[mjMAXUINAME];
    int bufi;

    int ncon = 0;
    for (int i = 0; i < d->ncon; ++i)
    {
        // dist <=0 is contact
        if (d->contact[i].dist > 0)
            continue;

        // first geom
        bufi = d->contact[i].geom1;   // first geom of the contact
        bufi = m->geom_bodyid[bufi];  // its body id
        mju_strncpy(buf, m->names+m->name_bodyadr[bufi], mjMAXUINAME);  // location of the name in buffer
        std::cout << "\t\tBody1 #" << bufi << " " << buf << " ";

        // same for the second geom
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[d->contact[i].geom2]], mjMAXUINAME);
        std::cout << "Body2 #" << m->geom_bodyid[d->contact[i].geom2] << " " << buf << " ";

        std::cout << "dist: " << d->contact[i].dist << " ";

        std::cout << std::endl;
    }
    if (ncon > 0)
        std::cout << "\tContacts detected: " << ncon << std::endl;
}


void color_touchpad_contacts(void)
{
    char buf[mjMAXUINAME];
    float* rgba;
    int bodygeom;
    int tpgeom;

    for (int i = 0; i < d->ncon; ++i)
    {
        if (d->contact[i].dist > 0)
            continue;

        tpgeom = -1;

        // print the text
        if (val_in_array(d->contact[i].geom1, num_touchpads, touchpad_geom_ids)) {
            tpgeom = d->contact[i].geom1;
            bodygeom = d->contact[i].geom2;
        }

        if (val_in_array(d->contact[i].geom2, num_touchpads, touchpad_geom_ids)) {
            if (tpgeom >= 0) {
                tpgeom = -1; // touching touchpad with touchpad, wtf
            }
            else {
                tpgeom = d->contact[i].geom2;
                bodygeom = d->contact[i].geom1;
            }
        }

        // color them
        if (tpgeom >= 0) {
            rgba = m->geom_rgba + 4*tpgeom;
            rgba[2] = 0.9999; //(float) i / d->ncon;
        }
    }
}


bool is_contacting(const int i_geom)
{
    bool answ = false;
    for (int j = 0; j < d->ncon; ++j)
        if ((d->contact[j].geom1 == i_geom || d->contact[j].geom2 == i_geom) && d->contact[j].dist <= 0) {
            answ = true;
            break;
        }
    return answ;
}


bool is_contacting_touchpad(const int i_geom)
{
    bool answ = false;
    char buf[mjMAXUINAME];
    int geom_potential_tp;

    // std::cout << "Testing contact of object " << i_geom;

    for (int j = 0; j < d->ncon; ++j) {
        geom_potential_tp = -1;
        if (d->contact[j].geom1 == i_geom)
            geom_potential_tp = d->contact[j].geom2;
        if (d->contact[j].geom2 == i_geom)
            geom_potential_tp = d->contact[j].geom1;

        // std::cout << " candidate " << geom_potential_tp << " " << d->contact[j].geom1 << " " << d->contact[j].geom2;

        if (val_in_array(geom_potential_tp, num_touchpads, touchpad_geom_ids) && d->contact[j].dist <= 0) {
            answ = true;
            break;
        }
    }

    // std::cout << std::endl;
    return answ;
}


bool is_contacting_hand(const int i)
{
    bool answ = false;
    char buf[mjMAXUINAME];
    int geom_potential_hand;

    for (int j = 0; j < d->ncon; ++j) {
        geom_potential_hand = -1;
        if (d->contact[j].geom1 == i)
            geom_potential_hand = d->contact[j].geom2;
        if (d->contact[j].geom2 == i)
            geom_potential_hand = d->contact[j].geom1;

        if (val_in_array(geom_potential_hand, num_hand_segs, hand_seg_geom_ids) && d->contact[j].dist <= 0) {
            answ = true;
            break;
        }
        if (answ)
            break;
    }
    return answ;
}


bool is_hand_touching_touchpad(void)
{
    bool answ = false;
    for (int i = 0; i < num_hand_segs; ++i)
        if (is_contacting_touchpad(hand_seg_geom_ids[i]))
        {
            answ = true;
            break;
        }
    return answ;
}


float depth_geom_pen_by_hand(const int i)
{
    float answ = 0.;
    int geom_potential_hand;

    for (int j = 0; j < d->ncon; ++j) {
        geom_potential_hand = -1;
        if (d->contact[j].geom1 == i)
            geom_potential_hand = d->contact[j].geom2;
        if (d->contact[j].geom2 == i)
            geom_potential_hand = d->contact[j].geom1;

        if (val_in_array(geom_potential_hand, num_hand_segs, hand_seg_geom_ids) && d->contact[j].dist < 0) {
            if (-d->contact[j].dist > answ)
                answ = -d->contact[j].dist;
        }
    }
    return answ;
}


float* contact_map(const int num_touchpads, const int* touchpad_geom_ids)
{
    float *contact_arr = new float[num_touchpads];
    int it;
    for (int i = 0; i < num_touchpads; ++i)
    {
        it = touchpad_geom_ids[i];
        if (is_contacting_hand(it))
            contact_arr[i] = 1;
        else
            contact_arr[i] = 0;
    }
    return contact_arr;
}


float* contact_map(void)
{
    float *contact_arr = new float[num_touchpads];
    int it;
    for (int i = 0; i < num_touchpads; ++i)
    {
        it = touchpad_geom_ids[i];
        if (is_contacting_hand(it))
            contact_arr[i] = 1;
        else
            contact_arr[i] = 0;
    }
    return contact_arr;
}


float* contact_dist_map(void)
{
    float *contact_arr = new float[num_touchpads];
    int it;
    for (int i = 0; i < num_touchpads; ++i)
    {
        it = touchpad_geom_ids[i];
        contact_arr[i] = depth_geom_pen_by_hand(i);
    }
    return contact_arr;
}


float resistive_segment_penetration(const int i)
{
    float answ = 0.;

    for (int j = 0; j < d->ncon; ++j) {
        // printf("%d %d %d %d\n", d->contact[j].geom1, d->contact[j].geom2, i, resistive_segment_geom_id);
        if (((d->contact[j].geom1 == i && d->contact[j].geom2 == resistive_segment_geom_id) ||
             (d->contact[j].geom2 == i && d->contact[j].geom1 == resistive_segment_geom_id)) &&
            d->contact[j].dist < 0) {
            answ += -d->contact[j].dist;
            // printf("touching answ %e\n", answ);
        }
    }
    return answ;
}


float resistive_segment_penetration_by_hand(void)
{
    float answ=0;
    for (int i = 0; i < num_hand_segs; ++i)
        answ += resistive_segment_penetration(hand_seg_geom_ids[i]);
    return answ;
}


int get_num_fingertip_contacts(void)
{
    int answ = 0;
    int g1, g2;
    std::vector<int> hfgis_l(hand_fingertip_geom_ids);

    for (int i = 0; i < d->ncon; ++i)
    {
        g1 = d->contact[i].geom1;
        g2 = d->contact[i].geom2;
        if (val_in_array(g1, num_touchpads, touchpad_geom_ids) && val_in_array(g2, hfgis_l))
        {
            answ++;
            hfgis_l.erase(std::find(hfgis_l.begin(), hfgis_l.end(), g2));
        }
        else if (val_in_array(g2, num_touchpads, touchpad_geom_ids) && val_in_array(g1, hfgis_l))
        {
            answ++;
            hfgis_l.erase(std::find(hfgis_l.begin(), hfgis_l.end(), g1));
        }
    }
    return answ;
}


void print_finger_touchpad_contacts(void)
{
    int g1, g2, g_s, g_t;

    for (int i = 0; i < d->ncon; ++i)
    {
        if (d->contact[i].dist > 0)
            continue;

        g1 = d->contact[i].geom1;
        g2 = d->contact[i].geom2;
        if (val_in_array(g1, num_touchpads, touchpad_geom_ids) && val_in_array(g2, num_hand_segs, hand_seg_geom_ids)) {
            g_s = g2;
            g_t = g1;
        }
        else if (val_in_array(g2, num_touchpads, touchpad_geom_ids) && val_in_array(g1, num_hand_segs, hand_seg_geom_ids)) {
            g_s = g1;
            g_t = g2;
        }
        else
            continue;

        std::cout << g1 << " " << g2 << " " << d->contact[i].pos[0] << " " << d->contact[i].pos[1] << " " << d->contact[i].pos[2] << " " << m->geom_rgba[4*g_t] << std::endl;
    }
}


//-------------------------------- coloring of touchpad ---------------------------------
float* get_geom_rgbas(void)
{
    float* geom_rgbas = new float[m->ngeom*4];
    for (int i = 0; i < m->ngeom*4; ++i)
        geom_rgbas[i] = m->geom_rgba[i];
    return geom_rgbas;
}


void set_touchpad_rgba(const float* geom_rgbas, const bool update_contacting=false)
{
    char buf[mjMAXUINAME];
    bool update;
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length())) {
            update = true;
            if (!update_contacting && is_contacting(i)) {
                update = false;
            }

            if (update) {
                m->geom_rgba[i*4] = geom_rgbas[i*4];
                m->geom_rgba[i*4+1] = geom_rgbas[i*4+1];
                m->geom_rgba[i*4+2] = geom_rgbas[i*4+2];
                m->geom_rgba[i*4+3] = geom_rgbas[i*4+3];
            }
            else {
                // m->geom_rgba[i*4+1] = 0.999;
            }
        }
    }
}


void print_who_is_blue(void)
{
    char buf[mjMAXUINAME];
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length())) {
            if ((m->geom_rgba[i*4+2] > 0) && !is_contacting(i)) {
                std::cout << "\t\t" << buf << " is blue " << m->geom_rgba[i*4+2] << " and not contacting" <<std::endl;
            }
        }
    }

    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, "TouchPad3D_14_00", 16)) {
            m->geom_rgba[i*4] = 1;
            for (int j = 0; j < m->ngeom; ++j)
            {
                if (m->geom_rgba[i*4+2] == m->geom_rgba[j*4+2] && m->geom_rgba[i*4+2] > 0 && i != j)
                {
                    mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[j]], mjMAXUINAME);
                    std::cout << "\t\t\tTouchPad3D_14_00 has same blue as " << buf << std::endl;
                }
            }
        }
    }
}


//-------------------------------- get and set global variables -------------------------
void get_touchpad_geom_ids(int* num_touchpads, int** touchpad_geom_ids)
{
    *num_touchpads = 0;

    char buf[mjMAXUINAME];
    // find how many
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length()))
            (*num_touchpads)++;
    }

    *touchpad_geom_ids = new int[*num_touchpads];
    int it=0;
    // fill the array
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length())) {
            (*touchpad_geom_ids)[it] = i;
            it++;
        }
    }
}


void set_touchpad_geom_ids(void)
{
    num_touchpads = 0;

    char buf[mjMAXUINAME];
    // find how many
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length()))
            num_touchpads++;
    }

    touchpad_geom_ids = new int[num_touchpads];
    int it=0;
    // fill the array
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BODY_NAME_PREFIX.c_str(), TOUCHPAD_BODY_NAME_PREFIX.length())) {
            touchpad_geom_ids[it] = i;
            it++;
        }
    }
}


void set_hand_geom_ids(void)
{
    num_hand_segs = 0;
    std::cout << "Counting hand geoms:";

    char buf[mjMAXUINAME];
    // find how many
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        for (int j = 0; j < HAND_SEGMENT_PREFIXES.size(); ++j)
            if (!strncmp(buf, HAND_SEGMENT_PREFIXES[j].c_str(), HAND_SEGMENT_PREFIXES[j].length())) {
                std::cout << " " << buf;
                num_hand_segs++;
                break;
            }
    }
    std::cout << std::endl;

    std::cout << "Adding hand geoms:";

    hand_seg_geom_ids = new int[num_hand_segs];
    int it = 0;
    // fill the array
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        for (int j = 0; j < HAND_SEGMENT_PREFIXES.size(); ++j)
            if (!strncmp(buf, HAND_SEGMENT_PREFIXES[j].c_str(), HAND_SEGMENT_PREFIXES[j].length())) {
                std::cout << " " << buf << " " << it << " " << i;
                hand_seg_geom_ids[it] = i;
                it++;
                break;
            }
    }
    std::cout << std::endl;
}


void get_segment_geom_ids(int* num_hand_segs, int** hand_seg_geom_ids)
{
    *num_hand_segs = 0;

    char buf[mjMAXUINAME];
    // find how many
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        for (int j = 0; j < HAND_SEGMENT_PREFIXES.size(); ++j)
            if (!strncmp(buf, HAND_SEGMENT_PREFIXES[j].c_str(), HAND_SEGMENT_PREFIXES[j].length())) {
                (*num_hand_segs)++;
                break;
            }
    }

    *hand_seg_geom_ids = new int[*num_hand_segs];
    int it=0;
    // fill the array
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        for (int j = 0; j < HAND_SEGMENT_PREFIXES.size(); ++j)
            if (!strncmp(buf, HAND_SEGMENT_PREFIXES[j].c_str(), HAND_SEGMENT_PREFIXES[j].length())) {
                (*hand_seg_geom_ids)[it] = i;
                it++;
                break;
            }
    }
}


void set_touchpad_manipulator_joints(void)
{
    num_touchpad_manipulator_joints = 0;

    char buf[mjMAXUINAME];
    // find how many
    for (int i = 0; i < m->njnt; ++i)
    {
        mju_strncpy(buf, m->names+m->name_jntadr[i], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_MANIPULATOR_JOINT_PREFIX.c_str(), TOUCHPAD_MANIPULATOR_JOINT_PREFIX.length()))
            num_touchpad_manipulator_joints++;
    }

    touchpad_manipulator_joints = new int[num_touchpad_manipulator_joints];
    int it=0;
    // fill the array
    for (int i = 0; i < m->njnt; ++i)
    {
        mju_strncpy(buf, m->names+m->name_jntadr[i], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_MANIPULATOR_JOINT_PREFIX.c_str(), TOUCHPAD_MANIPULATOR_JOINT_PREFIX.length())) {
            touchpad_manipulator_joints[it] = i;
            it++;
        }
    }
}


float* get_desired_contact_map(const float threshold, const int num_touchpads, const int* touchpad_geom_ids,
                               float* geom_rgbas)
{
    float *contact_arr = new float[num_touchpads];
    int it;
    float maxred = -1;

    // find the maximum red coloring
    for (int i = 0; i < num_touchpads; ++i)
    {
        it = touchpad_geom_ids[i];
        if (maxred < m->geom_rgba[it*4]) {
            maxred = m->geom_rgba[it*4];
        }
    }

    for (int i = 0; i < num_touchpads; ++i)
    {
        it = touchpad_geom_ids[i];
        if (m->geom_rgba[it*4] > maxred * threshold){
            contact_arr[i] = 1;
            geom_rgbas[it*4] = 0.9999;
        }
        else {
            contact_arr[i] = 0;
            geom_rgbas[it*4] = 0.0;
        }
    }

    return contact_arr;
}


float* get_desired_graded_contact_map(void)
{
    float *contact_arr = new float[num_touchpads];

    for (int i = 0; i < num_touchpads; ++i)
        contact_arr[i] = m->geom_rgba[touchpad_geom_ids[i]*4];

    return contact_arr;
}


void set_resistive_segment_geom_id(void)
{
    char buf[mjMAXUINAME];
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_bodyadr[m->geom_bodyid[i]], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_RESISTIVE_SEGMENT.c_str(), TOUCHPAD_RESISTIVE_SEGMENT.length()))
            resistive_segment_geom_id = i;
    }
}


void set_touchpad_base_geom_id(void)
{
    char buf[mjMAXUINAME];
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_geomadr[i], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_BASE_GEOM.c_str(), TOUCHPAD_BASE_GEOM.length()))
            touchpad_base_geom_id = i;
    }
}


void set_fingertip_geom_ids(void)
{

    char buf[mjMAXUINAME];

    std::cout << "Adding fingertip geoms:";
    // fill the array
    for (int i = 0; i < m->ngeom; ++i)
    {
        mju_strncpy(buf, m->names+m->name_geomadr[i], mjMAXUINAME);
        for (int j = 0; j < HAND_FINGERTIP_GEOMS.size(); ++j)
            if (!strncmp(buf, HAND_FINGERTIP_GEOMS[j].c_str(), HAND_FINGERTIP_GEOMS[j].length())) {
                std::cout << " " << buf << " " << i;
                hand_fingertip_geom_ids.push_back(i);
                break;
            }
    }
    std::cout << std::endl;
}


//-------------------------------- distances and postures -------------------------------
mjtNum* get_current_tpman_jointvals(void)
{
    mjtNum* answ = new mjtNum[num_touchpad_manipulator_joints];
    for (int i = 0; i < num_touchpad_manipulator_joints; ++i)
    {
        answ[i] = d->qpos[touchpad_manipulator_joints[i]];
    }
    return answ;
}


void set_current_tpman_jointvals(const mjtNum* x)
{
    for (int i = 0; i < num_touchpad_manipulator_joints; ++i)
        d->qpos[touchpad_manipulator_joints[i]] = x[i];
}


mjtNum hand_touchpad_distance(void)
{
    mjtNum answ = HAND_TOUCHPAD_DISTANCE_CAP;
    char buf[mjMAXUINAME];
    bool found=false;
    for (int i = 0; i < d->ncon; ++i)
    {
        mju_strncpy(buf, m->names+m->name_geomadr[d->contact[i].geom1], mjMAXUINAME);
        if (!strncmp(buf, TOUCHPAD_CENTREISH_GEOM.c_str(), TOUCHPAD_CENTREISH_GEOM.length())) {
            mju_strncpy(buf, m->names+m->name_geomadr[d->contact[i].geom2], mjMAXUINAME);
            if (!strncmp(buf, HAND_FINGER_GEOM.c_str(), HAND_FINGER_GEOM.length())) {
                found = true;
            }
        }
        else if (!strncmp(buf, HAND_FINGER_GEOM.c_str(), HAND_FINGER_GEOM.length())) {
            mju_strncpy(buf, m->names+m->name_geomadr[d->contact[i].geom2], mjMAXUINAME);
            if (!strncmp(buf, TOUCHPAD_CENTREISH_GEOM.c_str(), TOUCHPAD_CENTREISH_GEOM.length())) {
                found = true;
            }
        }

        if (found)
        {
            answ = d->contact[i].dist;
            break;
        }
    }
    return answ;
}


mjtNum fingertip_touchpad_distance(void)
{
    mjtNum answ = HAND_TOUCHPAD_DISTANCE_CAP;
    int g1, g2;

    for (int i = 0; i < d->ncon; ++i)
    {
        g1 = d->contact[i].geom1;
        g2 = d->contact[i].geom2;
        // std::cout<<g1 << " "<< g2 << " " << touchpad_base_geom_id;
        // for (int j = 0; j < hand_fingertip_geom_ids.size(); ++j)
        //     std::cout << " " << hand_fingertip_geom_ids[j];
        // std::cout << std::endl;
        if ((g1 == touchpad_base_geom_id && val_in_array(g2, hand_fingertip_geom_ids)) ||
            (g2 == touchpad_base_geom_id && val_in_array(g1, hand_fingertip_geom_ids))) {
            if (d->contact[i].dist < answ)
                answ = d->contact[i].dist;
        }
    }
    if (answ < 0)
        answ = 0;
    return answ;
}


/**
 * @brief Measures a sum of distance from the touchpad base to each fingertip
 * @details [long description]
 * @return [description]
 */
mjtNum fingertips_touchpad_distance(void)
{
    mjtNum answ = 0;
    int g1, g2;

    for (int i = 0; i < d->ncon; ++i)
    {
        g1 = d->contact[i].geom1;
        g2 = d->contact[i].geom2;
        if ((g1 == touchpad_base_geom_id && val_in_array(g2, hand_fingertip_geom_ids)) ||
            (g2 == touchpad_base_geom_id && val_in_array(g1, hand_fingertip_geom_ids))) {
            if (d->contact[i].dist > 0)
                answ += d->contact[i].dist;
        }
    }

    return answ;
}


float sensebox_hand_distance(const int isb)
{
    float answ = HAND_TOUCHPAD_DISTANCE_CAP;
    int g1, g2;
    for (int i = 0; i < d->ncon; ++i)
    {
        g1 = d->contact[i].geom1;
        g2 = d->contact[i].geom2;
        if ((g1 == isb && val_in_array(g2, num_hand_segs, hand_seg_geom_ids)) ||
            (g2 == isb && val_in_array(g1, num_hand_segs, hand_seg_geom_ids))) {
            if (d->contact[i].dist > 0)
                answ += (float) d->contact[i].dist;
            else if (d->contact[i].dist <= 0) {  // contact found
                answ = 0.;
                break;
            }
        }
    }

    return answ;
}


float* touchpad_boxes_hand_distance_map(void) {
    float *answ = new float[num_touchpads];
    for (int i = 0; i < num_touchpads; ++i)
        answ[i] = sensebox_hand_distance(touchpad_geom_ids[i]);
    return answ;
}


//-------------------------------- get cost function vars -------------------------------
float get_hand_touchpad_distance(void)
{
    // mjtNum answ = hand_touchpad_distance();
    mjtNum answ = fingertip_touchpad_distance();
    return (float) answ;
}


float get_contact_diff(void)
{
    // float* contact_arr = contact_map();
    float* contact_arr = contact_dist_map();
    normalize_arr(contact_arr, num_touchpads);
    return rms(num_touchpads, desired_contact_map, contact_arr);
}


float get_resistive_contact(void)
{
    return resistive_segment_penetration_by_hand();
}


int get_num_fingertips_touchpad(void)
{
    return get_num_fingertip_contacts();
}


float get_fingertips_touchpad_distance(void)
{
    return fingertips_touchpad_distance();
}


bool get_is_hand_touching_touchpad(void)
{
    return is_hand_touching_touchpad();
}


float get_touchpad_hand_magnetism(void)
{
    float threshold = 0.1;
    float *dcm_thresholded = new float(num_touchpads);
    for (int i = 0; i < num_touchpads; ++i)
        if (desired_contact_map[i] > threshold)
            dcm_thresholded[i] = desired_contact_map[i];
        else
            dcm_thresholded[i] = 0.;
    return scalar_multiplication(num_touchpads, dcm_thresholded, touchpad_boxes_hand_distance_map());
}



//-------------------------------- runtime variables ------------------------------------
void update_runtime_hand_touchpad_distance(void)
{
    runtime_hand_touchpad_distance = get_hand_touchpad_distance();
}


void update_runtime_contact_diff(void)
{
    runtime_contact_diff = get_contact_diff();
}


void update_runtime_resistive_contact(void)
{
    runtime_resseg_hand_penetration = get_resistive_contact();
}


void update_runtime_num_fingertips_touchpad(void)
{
    runtime_num_fingertips_touchpad = get_num_fingertips_touchpad();
}


void update_runtime_fingertips_touchpad_distance(void)
{
    runtime_fingertips_touchpad_distance = get_fingertips_touchpad_distance();
}


void update_runtime_is_hand_touching_touchpad(void)
{
    runtime_hand_is_touching = is_hand_touching_touchpad();
}


void update_runtime_touchpad_hand_magnetism(void)
{
    runtime_touchpad_hand_magnetism = get_touchpad_hand_magnetism();
}


/**
 * @brief updates all components of the runtime metric
 * @details [long description]
 */
void update_runtime_metric(void)
{
    // apply pose perturbation
    mjv_applyPerturbPose(m, d, &pert, 1);      // move mocap and dynamic bodies

    // run mj_forward, to update rendering and joint sliders
    mj_forward(m, d);

    runtime_contact_diff = get_contact_diff();
    // runtime_hand_touchpad_distance = get_hand_touchpad_distance();
    runtime_resseg_hand_penetration = get_resistive_contact();
    runtime_fingertips_touchpad_distance = get_fingertips_touchpad_distance();
    runtime_num_fingertips_touchpad = get_num_fingertips_touchpad();
    runtime_hand_is_touching = get_is_hand_touching_touchpad();
    runtime_touchpad_hand_magnetism = get_touchpad_hand_magnetism();

    runtime_metric = COST_FUNCTION_CONTACTMAP_WEIGHT*runtime_contact_diff;

    runtime_metric += COST_FUNCTION_PENETRATION_WEIGHT*(pow(runtime_resseg_hand_penetration, 2));

    runtime_metric += COST_FUNCTION_MAGNETISM_WEIGHT*runtime_touchpad_hand_magnetism;

    if (runtime_num_fingertips_touchpad < desired_num_fingertip_contacts)
        runtime_metric += COST_FUNCTION_FINGERTIPS_WEIGHT * runtime_fingertips_touchpad_distance;

    // if (!runtime_hand_is_touching)
    //     runtime_metric += (runtime_hand_touchpad_distance+1) * COST_FUNCTION_DISTANCE_WEIGHT;
}


/**
 * @brief for background simulation process
 * @details [long description]
 */
void reliable_update_runtime_metric(void)
{
    // apply pose perturbation
    mjv_applyPerturbPose(m, d, &pert, 1);      // move mocap and dynamic bodies

    // run mj_forward, to update rendering and joint sliders
    mj_forward(m, d);

    std::vector<float> rcd_v, rsp_v, rhtd_v, rftd_v, rthm_v;
    std::vector<int> rnft_v;
    std::vector<bool> hit_v;

    rcd_v.push_back(get_contact_diff());
    // rsp_v.push_back(get_hand_touchpad_distance());
    rhtd_v.push_back(get_resistive_contact());
    rftd_v.push_back(get_num_fingertips_touchpad());
    rnft_v.push_back(get_fingertips_touchpad_distance());
    hit_v.push_back(get_is_hand_touching_touchpad());
    rthm_v.push_back(get_touchpad_hand_magnetism());

    for (int i = 0; i < 0; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        rcd_v.push_back(get_contact_diff());
        // rsp_v.push_back(get_hand_touchpad_distance());
        rhtd_v.push_back(get_resistive_contact());
        rftd_v.push_back(get_num_fingertips_touchpad());
        rnft_v.push_back(get_fingertips_touchpad_distance());
        hit_v.push_back(get_is_hand_touching_touchpad());
        rthm_v.push_back(get_touchpad_hand_magnetism());
    }

    runtime_contact_diff =                  median(rcd_v);
    // runtime_hand_touchpad_distance =        median(rsp_v);
    runtime_resseg_hand_penetration =       median(rhtd_v);
    runtime_fingertips_touchpad_distance =  median(rftd_v);
    runtime_num_fingertips_touchpad =       median(rnft_v);
    runtime_hand_is_touching =              median(hit_v);
    runtime_touchpad_hand_magnetism =       median(rthm_v);

    runtime_metric = COST_FUNCTION_CONTACTMAP_WEIGHT*runtime_contact_diff;

    runtime_metric += COST_FUNCTION_PENETRATION_WEIGHT*(pow(runtime_resseg_hand_penetration, 2));

    runtime_metric += COST_FUNCTION_MAGNETISM_WEIGHT*runtime_touchpad_hand_magnetism;

    if (runtime_num_fingertips_touchpad < desired_num_fingertip_contacts)
        runtime_metric += COST_FUNCTION_FINGERTIPS_WEIGHT * runtime_fingertips_touchpad_distance;

    // if (!runtime_hand_is_touching)
    //     runtime_metric += (runtime_hand_touchpad_distance+1) * COST_FUNCTION_DISTANCE_WEIGHT;
}



//-------------------------------- cost function ----------------------------------------
float cost_function(void)
{
    // mj_forward(m, d);
    settings.run = 0;
    // start exclusive access (block simulation thread)
    mtx.lock();

    // handle events (calls all callbacks)
    glfwPollEvents();

    // prepare to render
    prepare();

    // end exclusive access (allow simulation thread to run)
    mtx.unlock();

    // render while simulation is running
    render(window);

    // update_runtime_contact_diff();
    // update_runtime_hand_touchpad_distance();
    // update_runtime_resistive_contact();  // happens in update_runtime_metric
    update_runtime_metric();

    // reliable_update_runtime_metric();

    return runtime_metric;
}


float cost_function(const mjtNum* x)
{
    set_current_tpman_jointvals(x);

    cost_function();

    return runtime_metric;
}



//-------------------------------- nlopt wrappers ---------------------------------------
typedef struct {
    bool print_cost_function_calls = false;
} CF_DATA;


double cost_function_nlopt_wr(unsigned n, const double *x, double *grad, void *my_func_data)
{
    CF_DATA *d = (CF_DATA *) my_func_data;

    float answ = cost_function(x);

    if (d->print_cost_function_calls) {
        std::cout << "\t\tCost function call #" << std::setw(8) << how_many_times_you_called_me << ", ";
        printf("%7.4f%% done. x = [", 100.0*how_many_times_you_called_me/COST_FUNCTION_GLOBAL_EVAL);
        for (int i = 0; i < n; ++i)
            printf(" %10.6f", x[i]);
            // std::cout << " " << std::setw(8) << std::setprecision(4) << x[i];
        printf("] f = %10.6f bestf = %10.6f on it#%d\r", answ, min_cf_val, iteration_when_min_was_found);
        // std::cout << "] f= " << std::setw(8) << std::setprecision(4) << answ << "\r";
    }

    how_many_times_you_called_me++;

    if (answ < min_cf_val)
    {
        min_cf_val = answ;
        iteration_when_min_was_found = how_many_times_you_called_me-1;
        for (int i = 0; i < n; ++i)
            min_cf_pose[i] = x[i];
    }

    return (double) answ;
}



//-------------------------------- nlopt testing ----------------------------------------
double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    if (grad) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}


typedef struct {
    double a, b;
} my_constraint_data;


double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
    my_constraint_data *d = (my_constraint_data *) data;
    double a = d->a, b = d->b;
    if (grad) {
        grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
        grad[1] = -1.0;
    }
    return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}




//-------------------------------- init and main ----------------------------------------

// initalize everything
void init(void)
{
    // print version, check compatibility
    printf("MuJoCo Pro version %.2lf\n", 0.01*mj_version());
    if( mjVERSION_HEADER!=mj_version() )
        mju_error("Headers and library have different versions");

    // activate MuJoCo license
    mj_activate("mjkey.txt");

    // init GLFW, set timer callback (milliseconds)
    if (!glfwInit())
        mju_error("could not initialize GLFW");
    mjcb_time = timer;

    // multisampling
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_VISIBLE, 1);

    // get videomode and save
    vmode = *glfwGetVideoMode(glfwGetPrimaryMonitor());

    // create window
    window = glfwCreateWindow((2*vmode.width)/3, (2*vmode.height)/3,
                              "Simulate", NULL, NULL);
    if( !window )
    {
        glfwTerminate();
        mju_error("could not create window");
    }

    // save window position and size
    glfwGetWindowPos(window, windowpos, windowpos+1);
    glfwGetWindowSize(window, windowsize, windowsize+1);

    // make context current, set v-sync
    glfwMakeContextCurrent(window);
    glfwSwapInterval(settings.vsync);

    // init abstract visualization
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    profilerinit();
    sensorinit();

    // make empty scene
    mjv_defaultScene(&scn);
    mjv_makeScene(NULL, &scn, maxgeom);

    // select default font
    int fontscale = uiFontScale(window);
    settings.font = fontscale/50 - 1;

    // make empty context
    mjr_defaultContext(&con);
    mjr_makeContext(NULL, &con, fontscale);

    // set GLFW callbacks
    uiSetCallback(window, &uistate, uiEvent, uiLayout);
    glfwSetWindowRefreshCallback(window, render);
    glfwSetDropCallback(window, drop);

    // init state and uis
    memset(&uistate, 0, sizeof(mjuiState));
    memset(&ui0, 0, sizeof(mjUI));
    memset(&ui1, 0, sizeof(mjUI));
    ui0.spacing = mjui_themeSpacing(settings.spacing);
    ui0.color = mjui_themeColor(settings.color);
    ui0.predicate = uiPredicate;
    ui0.rectid = 1;
    ui0.auxid = 0;
    ui1.spacing = mjui_themeSpacing(settings.spacing);
    ui1.color = mjui_themeColor(settings.color);
    ui1.predicate = uiPredicate;
    ui1.rectid = 2;
    ui1.auxid = 1;

    // populate uis with standard sections
    mjui_add(&ui0, defFile);
    mjui_add(&ui0, defOption);
    mjui_add(&ui0, defSimulation);
    mjui_add(&ui0, defWatch);
    uiModify(window, &ui0, &uistate, &con);
    uiModify(window, &ui1, &uistate, &con);
}



// run event loop
int main(int argc, const char** argv)
{
    // initialize everything
    init();

    printf("This is simulate adjusted version 1.0.9\n");

    //--------------- always load model
    if (argc < 2) {
            sprintf(filename, "%s/touch_pad_deeplabcut_rig/inverse_kinematics/mjc_model/Hand_18_DOF_meshes_att_touchpad3d.xml",
                    getenv("BL_REPOSITORIES"));
    } else if (argc < 3) {
        if (!strcmp(argv[1], "default") || !strcmp(argv[1], "short"))
            sprintf(filename, "%s/touch_pad_deeplabcut_rig/inverse_kinematics/mjc_model/Hand_18_DOF_meshes_att_touchpad3d.xml",
                    getenv("BL_REPOSITORIES"));
        else if (!strcmp(argv[1], "nomesh") || !strcmp(argv[1], "no_mesh") || !strcmp(argv[1], "nomeshes") || !strcmp(argv[1], "no_meshes"))
            sprintf(filename, "%s/touch_pad_deeplabcut_rig/inverse_kinematics/mjc_model/Hand_18_DOF_att_touchpad3d.xml",
                    getenv("BL_REPOSITORIES"));
        else
            strcpy(filename, argv[1]);
    } else {
        if (!strcmp(argv[1], "short"))
            sprintf(filename, "%s/touch_pad_deeplabcut_rig/inverse_kinematics/mjc_model/Hand_18_DOF_%s_att_touchpad3d.xml",
                    getenv("BL_REPOSITORIES"), argv[2]);
    }
    settings.loadrequest = 0;
    settings.run = 0;
    loadmodel();

    // start simulation thread
    // std::thread simthread(simulate);


    //-------------- load kinematic data
    int M, N;
    mjtNum *nTime, *nPos;

    std::vector<std::string> dof_names=load_mot_file(
        "C:/FLIR_cameras/PublicExample/exp_session_2019.12.20_11.11.21_AS_CMG_11_subset/8cams11subset-AS-2020-01-04/inverse_kinematics/out_inv_kin_arm_and_touchpad.mot", &M, &N, &nTime, &nPos);

    mjtNum *nPosAverage = average_position(nPos, M, N);
    std::vector<int> dof_map=map_dofs(dof_names);
    std::vector<mjtNum> dof_multiplier(M);
    std::vector<mjtNum> dof_offset(M);
    for (int i = 0; i < M; ++i)
    {
        dof_multiplier[i] = mjPI/180;
        dof_offset[i] = 0;
    }
    dof_offset[18] = mjPI/2;

    assume_posture(M, dof_map, nPos+M, dof_multiplier, dof_offset);
    float* geom_rgbas=get_geom_rgbas();


    //------------------ set globals
    set_touchpad_geom_ids();
    set_touchpad_manipulator_joints();
    set_hand_geom_ids();
    set_resistive_segment_geom_id();
    set_touchpad_base_geom_id();
    set_fingertip_geom_ids();

    // std::cout << "Touchpad geom ids:";
    // for (int i = 0; i < num_touchpads; ++i)
    // {
    //     std::cout << " " << touchpad_geom_ids[i];
    // }
    // std::cout << std::endl;
    // std::cout << "Hand geom ids:";
    // for (int i = 0; i < num_hand_segs; ++i)
    // {
    //     std::cout << " " << hand_seg_geom_ids[i];
    // }
    // std::cout << std::endl;


    // desired_contact_map = get_desired_contact_map(0.1, num_touchpads, touchpad_geom_ids, geom_rgbas);
    desired_contact_map = get_desired_graded_contact_map();
    normalize_arr(desired_contact_map, num_touchpads);
    float* contact_arr;
    float diff_contact, dist_contact;
    char buf[mjMAXUINAME];
    // initial_contact_diff = (float) hand_touchpad_distance();


    //------------------ optimization presetup
    mj_forward(m, d);

    CF_DATA *cf_data = new CF_DATA;
    cf_data->print_cost_function_calls = true;

    // double *x = get_current_tpman_jointvals();
    double x[] = {1.54587, 0.21131, -2.25106, -0.319758, -0.984643, 0.433356};
    // double x[] = {1.58217, 0.169781, -2.27062, -0.251719, -0.827643, 0.590356};
    set_current_tpman_jointvals(x);

    double *lb = new double[num_touchpad_manipulator_joints];
    double *ub = new double[num_touchpad_manipulator_joints];
    double range_width;
    for (int i = 0; i < num_touchpad_manipulator_joints; ++i)
    {
        range_width = m->jnt_range[2*touchpad_manipulator_joints[i]+1] - m->jnt_range[2*touchpad_manipulator_joints[i]];
        lb[i] = x[i] - range_width*0.05;
        ub[i] = x[i] + range_width*0.05;
    }


    //------------------ optimization setup
    nlopt_opt opt, locopt;

    // local optimizer
    locopt = nlopt_create(NLOPT_LN_SBPLX, num_touchpad_manipulator_joints);
    // locopt = nlopt_create(NLOPT_LD_MMA, num_touchpad_manipulator_joints);
    nlopt_set_min_objective(locopt, cost_function_nlopt_wr, cf_data);

    nlopt_set_lower_bounds(locopt, lb);
    nlopt_set_upper_bounds(locopt, ub);

    nlopt_set_initial_step1(locopt, 0.1);

    nlopt_set_ftol_rel(locopt, 1e-2);
    nlopt_set_xtol_rel(locopt, 1e-2);

    nlopt_set_maxeval(locopt, COST_FUNCTION_LOCAL_EVAL);
    // global optimizer
    opt = nlopt_create(NLOPT_G_MLSL_LDS, num_touchpad_manipulator_joints);
    nlopt_set_min_objective(opt, cost_function_nlopt_wr, cf_data);

    nlopt_set_local_optimizer(opt, locopt);

    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    nlopt_set_ftol_rel(opt, 1e-2);
    nlopt_set_xtol_rel(opt, 1e-2);

    nlopt_set_initial_step1(opt, 0.1);

    nlopt_set_maxeval(opt, COST_FUNCTION_GLOBAL_EVAL);

    min_cf_pose = new double[num_touchpad_manipulator_joints]; // saves best poses

    //------------------ optimization
    double minf = cost_function_nlopt_wr(num_touchpad_manipulator_joints, x, NULL, cf_data);
    int res = -1;

    res = nlopt_optimize(opt, x, &minf);

    std::cout << std::endl;
    if (res < 0) {
        std::cout << "Nlopt failed with code " << res << std::endl;
    }
    else {
        std::cout << "Found minimum at x = [";
        for (int i = 0; i < num_touchpad_manipulator_joints; ++i)
            std::cout << " " << x[i];
        std::cout << "] f = " << minf << std::endl;
    }

    std::cout << "Stored minimum at x = [";
    for (int i = 0; i < num_touchpad_manipulator_joints; ++i)
        std::cout << " " << min_cf_pose[i];
    std::cout << "] f = " << min_cf_val << " found on iteration " << iteration_when_min_was_found << std::endl;

    set_current_tpman_jointvals(min_cf_pose);

    nlopt_destroy(locopt);
    nlopt_destroy(opt);

    print_finger_touchpad_contacts();

    //-------------------------- event loop
    while( !glfwWindowShouldClose(window) && !settings.exitrequest )
    {
        // std::cout << "----------------------------------------------" << std::endl;
        settings.run = 0;
        // start exclusive access (block simulation thread)
        mtx.lock();

        // handle events (calls all callbacks)
        glfwPollEvents();

        // prepare to render
        prepare();

        // end exclusive access (allow simulation thread to run)
        mtx.unlock();

        // render while simulation is running
        render(window);

        set_touchpad_rgba(geom_rgbas, true);
        color_touchpad_contacts();
        // print_contacts();
        // print_who_is_blue();

        // update_runtime_contact_diff();
        // update_runtime_hand_touchpad_distance();
        // cost_function();
        update_runtime_metric();
        // reliable_update_runtime_metric();

        // std::cout << "RMS difference: " << std::setprecision(12) << runtime_contact_diff << " ";
        // std::cout << "Runtime hand to TP distance: " << std::setprecision(12) << runtime_hand_touchpad_distance << " ";
        // std::cout << "Total metric: " << std::setprecision(12) << runtime_metric << "\r";
        printf("RMS difference: %10.6f", runtime_contact_diff);
        // printf(" hand to TP distance: %10.6f", runtime_hand_touchpad_distance);
        printf(" touching resseg: %10.6f", runtime_resseg_hand_penetration);
        printf(" TP magnetism: %10.6f", runtime_touchpad_hand_magnetism);
        printf(" numfing touching tp %2d FT TP dist %10.6f",
               runtime_num_fingertips_touchpad, runtime_fingertips_touchpad_distance);

        printf(" hand is");
        if (!runtime_hand_is_touching)
            printf(" NOT");
        printf(" touching TP. Total metric: %.6f\r", runtime_metric);

        if (print_current_hand_touchpad_contacts)
        {
            print_finger_touchpad_contacts();
            print_current_hand_touchpad_contacts = 0;
        }

        // settings.exitrequest = 1; ///////////////////////////////
    }

    std::cout << std::endl;

    // stop simulation thread
    settings.exitrequest = 1;
    // simthread.join();

    // delete everything we allocated
    uiClearCallback(window);
    mj_deleteData(d);
    mj_deleteModel(m);
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // deactive MuJoCo
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 0;
}
