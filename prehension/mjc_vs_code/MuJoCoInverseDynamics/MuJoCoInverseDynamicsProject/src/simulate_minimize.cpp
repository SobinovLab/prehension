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
#include <iostream>
#include <filesystem>
#include <gl/gl.h>
#include <filesystem>
#include <math.h>
#include <iterator>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <algorithm>
#include <limits>
#include <map>
#include <regex>
#include <atomic>
#include <opencv2/opencv.hpp>

// using C API
#include <nlopt.h>
#include "Basinhopping.h"

#include "IOFunctions.h"
#include "MiscArrayFunctions.h"

#include "FrameSavingBuffer.h"

#pragma comment(lib, "OpenGL32.lib")

namespace fs = std::filesystem;

// visual/sim thread synchronization
std::mutex mtx;

// declare for GUI use
void assume_posture_and_print_cost(bool lock_mutex = true);
void export_current_adjustment(const double* x, const int n);

// final posture
int n_x_final = 0;
double* x_final = NULL;
double* x_step = NULL;

std::string adjustment_file;

//---------------------------------------------------------------------------------------
//-------------------------------- BASIC SIMULATE ---------------------------------------
//---------------------------------------------------------------------------------------

//-------------------------------- global -----------------------------------------------

// constants
const int maxgeom = 10000;           // preallocated geom array in mjvScene
const double syncmisalign = 0.1;    // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7;   // fraction of refresh available for simulation

// model and data
mjModel* m = NULL;
mjData* d = NULL;

bool visuals_enabled = true;
std::atomic<int> verbose = 0;

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
    int run = 0;
    int key = 0;
    int loadrequest = 0;

    int done_optimizing = 0;

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
    SECT_FILE = 0,
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
    figconstraint.figurergba[0] = 0.1f;
    figcost.figurergba[2] = 0.2f;
    figsize.figurergba[0] = 0.1f;
    figtimer.figurergba[2] = 0.2f;
    figconstraint.figurergba[3] = 0.5f;
    figcost.figurergba[3] = 0.5f;
    figsize.figurergba[3] = 0.5f;
    figtimer.figurergba[3] = 0.5f;

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
    for (n = 0; n < 6; n++)
        for (i = 0; i < mjMAXLINEPNT; i++)
        {
            figtimer.linedata[n][2 * i] = (float)-i;
            figsize.linedata[n][2 * i] = (float)-i;
        }
}



// update profiler figures
void profilerupdate(void)
{
    int i, n;

    // update constraint figure
    figconstraint.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
    for (i = 1; i < 5; i++)
        figconstraint.linepnt[i] = figconstraint.linepnt[0];
    if (m->opt.solver == mjSOL_PGS)
    {
        figconstraint.linepnt[3] = 0;
        figconstraint.linepnt[4] = 0;
    }
    if (m->opt.solver == mjSOL_CG)
        figconstraint.linepnt[4] = 0;
    for (i = 0; i < figconstraint.linepnt[0]; i++)
    {
        // x
        figconstraint.linedata[0][2 * i] = (float)i;
        figconstraint.linedata[1][2 * i] = (float)i;
        figconstraint.linedata[2][2 * i] = (float)i;
        figconstraint.linedata[3][2 * i] = (float)i;
        figconstraint.linedata[4][2 * i] = (float)i;

        // y
        figconstraint.linedata[0][2 * i + 1] = (float)d->nefc;
        figconstraint.linedata[1][2 * i + 1] = (float)d->solver[i].nactive;
        figconstraint.linedata[2][2 * i + 1] = (float)d->solver[i].nchange;
        figconstraint.linedata[3][2 * i + 1] = (float)d->solver[i].neval;
        figconstraint.linedata[4][2 * i + 1] = (float)d->solver[i].nupdate;
    }

    // update cost figure
    figcost.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
    for (i = 1; i < 3; i++)
        figcost.linepnt[i] = figcost.linepnt[0];
    if (m->opt.solver == mjSOL_PGS)
    {
        figcost.linepnt[1] = 0;
        figcost.linepnt[2] = 0;
    }

    for (i = 0; i < figcost.linepnt[0]; i++)
    {
        // x
        figcost.linedata[0][2 * i] = (float)i;
        figcost.linedata[1][2 * i] = (float)i;
        figcost.linedata[2][2 * i] = (float)i;

        // y
        figcost.linedata[0][2 * i + 1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].improvement));
        figcost.linedata[1][2 * i + 1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].gradient));
        figcost.linedata[2][2 * i + 1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].lineslope));
    }

    // get timers: total, collision, prepare, solve, other
    mjtNum total = d->timer[mjTIMER_STEP].duration;
    int number = d->timer[mjTIMER_STEP].number;
    if (!number)
    {
        total = d->timer[mjTIMER_FORWARD].duration;
        number = d->timer[mjTIMER_FORWARD].number;
    }
    number = mjMAX(1, number);
    float tdata[5] = {
        (float)(total / number),
        (float)(d->timer[mjTIMER_POS_COLLISION].duration / number),
        (float)(d->timer[mjTIMER_POS_MAKE].duration / number) +
            (float)(d->timer[mjTIMER_POS_PROJECT].duration / number),
        (float)(d->timer[mjTIMER_CONSTRAINT].duration / number),
        0
    };
    tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

    // update figtimer
    int pnt = mjMIN(201, figtimer.linepnt[0] + 1);
    for (n = 0; n < 5; n++)
    {
        // shift data
        for (i = pnt - 1; i > 0; i--)
            figtimer.linedata[n][2 * i + 1] = figtimer.linedata[n][2 * i - 1];

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
    pnt = mjMIN(201, figsize.linepnt[0] + 1);
    for (n = 0; n < 6; n++)
    {
        // shift data
        for (i = pnt - 1; i > 0; i--)
            figsize.linedata[n][2 * i + 1] = figsize.linedata[n][2 * i - 1];

        // assign new
        figsize.linepnt[n] = pnt;
        figsize.linedata[n][1] = sdata[n];
    }
}



// show profiler figures
void profilershow(mjrRect rect)
{
    mjrRect viewport = {
        rect.left + rect.width - rect.width / 4,
        rect.bottom,
        rect.width / 4,
        rect.height / 4
    };
    mjr_figure(viewport, &figtimer, &con);
    viewport.bottom += rect.height / 4;
    mjr_figure(viewport, &figsize, &con);
    viewport.bottom += rect.height / 4;
    mjr_figure(viewport, &figcost, &con);
    viewport.bottom += rect.height / 4;
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
    for (int i = 0; i < maxline; i++)
        figsensor.linepnt[i] = 0;

    // start with line 0
    int lineid = 0;

    // loop over sensors
    for (int n = 0; n < m->nsensor; n++)
    {
        // go to next line if type is different
        if (n > 0 && m->sensor_type[n] != m->sensor_type[n - 1])
            lineid = mjMIN(lineid + 1, maxline - 1);

        // get info about this sensor
        mjtNum cutoff = (m->sensor_cutoff[n] > 0 ? m->sensor_cutoff[n] : 1);
        int adr = m->sensor_adr[n];
        int dim = m->sensor_dim[n];

        // data pointer in line
        int p = figsensor.linepnt[lineid];

        // fill in data for this sensor
        for (int i = 0; i < dim; i++)
        {
            // check size
            if ((p + 2 * i) >= mjMAXLINEPNT / 2)
                break;

            // x
            figsensor.linedata[lineid][2 * p + 4 * i] = (float)(adr + i);
            figsensor.linedata[lineid][2 * p + 4 * i + 2] = (float)(adr + i);

            // y
            figsensor.linedata[lineid][2 * p + 4 * i + 1] = 0;
            figsensor.linedata[lineid][2 * p + 4 * i + 3] = (float)(d->sensordata[adr + i] / cutoff);
        }

        // update linepnt
        figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT - 1,
            figsensor.linepnt[lineid] + 2 * dim);
    }
}



// show sensor figure
void sensorshow(mjrRect rect)
{
    // constant width with and without profiler
    int width = settings.profiler ? rect.width / 3 : rect.width / 4;

    // render figure on the right
    mjrRect viewport = {
        rect.left + rect.width - width,
        rect.bottom,
        width,
        rect.height / 3
    };
    mjr_figure(viewport, &figsensor, &con);
}



// prepare info text
void infotext(char* title, char* content, double interval)
{
    char tmp[20];

    // compute solver error
    mjtNum solerr = 0;
    if (d->solver_iter)
    {
        int ind = mjMIN(d->solver_iter - 1, mjNSOLVER - 1);
        solerr = mju_min(d->solver[ind].improvement, d->solver[ind].gradient);
        if (solerr == 0)
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
        1 / interval,
        d->maxuse_stack / (double)d->nstack,
        d->maxuse_con / (double)m->nconmax,
        d->maxuse_efc / (double)m->njmax);

    // add Energy if enabled
    if (mjENABLED(mjENBL_ENERGY))
    {
        sprintf(tmp, "\n%.3f", d->energy[0] + d->energy[1]);
        strcat(content, tmp);
        strcat(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if (mjENABLED(mjENBL_FWDINV))
    {
        sprintf(tmp, "\n%.1f %.1f",
            mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[0])),
            mju_log10(mju_max(mjMINVAL, d->solver_fwdinv[1])));
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
    for (i = 0; i < mjNDISABLE; i++)
    {
        strcpy(defFlag[0].name, mjDISABLESTRING[i]);
        defFlag[0].pdata = settings.disable + i;
        mjui_add(&ui0, defFlag);
    }
    mjui_add(&ui0, defEnableFlags);
    for (i = 0; i < mjNENABLE; i++)
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
    for (i = 0; i < mjMIN(m->ncam, mjMAXUIMULTI - 2); i++)
    {
        // prepare name
        char camname[mjMAXUITEXT] = "\n";
        if (m->names[m->name_camadr[i]])
            strcat(camname, m->names + m->name_camadr[i]);
        else
            sprintf(camname, "\nCamera %d", i);

        // check string length
        if (strlen(camname) + strlen(defRendering[1].other) >= mjMAXUITEXT - 1)
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
    for (i = 0; i < mjNVISFLAG; i++)
    {
        // set name, remove "&"
        strcpy(defFlag[0].name, mjVISSTRING[i][0]);
        for (j = 0; j < strlen(mjVISSTRING[i][0]); j++)
            if (mjVISSTRING[i][0][j] == '&')
            {
                strcpy(defFlag[0].name + j, mjVISSTRING[i][0] + j + 1);
                break;
            }

        // set shortcut and data
        sprintf(defFlag[0].other, " %s", mjVISSTRING[i][2]);
        defFlag[0].pdata = vopt.flags + i;
        mjui_add(&ui0, defFlag);
    }
    mjui_add(&ui0, defOpenGL);
    for (i = 0; i < mjNRNDFLAG; i++)
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
        {mjITEM_CHECKBYTE,  "Geom 1",           2, vopt.geomgroup + 1,        " 1"},
        {mjITEM_CHECKBYTE,  "Geom 2",           2, vopt.geomgroup + 2,        " 2"},
        {mjITEM_CHECKBYTE,  "Geom 3",           2, vopt.geomgroup + 3,        " 3"},
        {mjITEM_CHECKBYTE,  "Geom 4",           2, vopt.geomgroup + 4,        " 4"},
        {mjITEM_CHECKBYTE,  "Geom 5",           2, vopt.geomgroup + 5,        " 5"},
        {mjITEM_SEPARATOR,  "Site groups",  1},
        {mjITEM_CHECKBYTE,  "Site 0",           2, vopt.sitegroup,          "S0"},
        {mjITEM_CHECKBYTE,  "Site 1",           2, vopt.sitegroup + 1,        "S1"},
        {mjITEM_CHECKBYTE,  "Site 2",           2, vopt.sitegroup + 2,        "S2"},
        {mjITEM_CHECKBYTE,  "Site 3",           2, vopt.sitegroup + 3,        "S3"},
        {mjITEM_CHECKBYTE,  "Site 4",           2, vopt.sitegroup + 4,        "S4"},
        {mjITEM_CHECKBYTE,  "Site 5",           2, vopt.sitegroup + 5,        "S5"},
        {mjITEM_SEPARATOR,  "Joint groups", 1},
        {mjITEM_CHECKBYTE,  "Joint 0",          2, vopt.jointgroup,         ""},
        {mjITEM_CHECKBYTE,  "Joint 1",          2, vopt.jointgroup + 1,       ""},
        {mjITEM_CHECKBYTE,  "Joint 2",          2, vopt.jointgroup + 2,       ""},
        {mjITEM_CHECKBYTE,  "Joint 3",          2, vopt.jointgroup + 3,       ""},
        {mjITEM_CHECKBYTE,  "Joint 4",          2, vopt.jointgroup + 4,       ""},
        {mjITEM_CHECKBYTE,  "Joint 5",          2, vopt.jointgroup + 5,       ""},
        {mjITEM_SEPARATOR,  "Tendon groups",    1},
        {mjITEM_CHECKBYTE,  "Tendon 0",         2, vopt.tendongroup,        ""},
        {mjITEM_CHECKBYTE,  "Tendon 1",         2, vopt.tendongroup + 1,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 2",         2, vopt.tendongroup + 2,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 3",         2, vopt.tendongroup + 3,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 4",         2, vopt.tendongroup + 4,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 5",         2, vopt.tendongroup + 5,      ""},
        {mjITEM_SEPARATOR,  "Actuator groups", 1},
        {mjITEM_CHECKBYTE,  "Actuator 0",       2, vopt.actuatorgroup,      ""},
        {mjITEM_CHECKBYTE,  "Actuator 1",       2, vopt.actuatorgroup + 1,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 2",       2, vopt.actuatorgroup + 2,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 3",       2, vopt.actuatorgroup + 3,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 4",       2, vopt.actuatorgroup + 4,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 5",       2, vopt.actuatorgroup + 5,    ""},
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
    for (i = 0; i < m->njnt && itemcnt < mjMAXUIITEM; i++)
        if ((m->jnt_type[i] == mjJNT_HINGE || m->jnt_type[i] == mjJNT_SLIDE))
        {
            // skip if joint group is disabled
            if (!vopt.jointgroup[mjMAX(0, mjMIN(mjNGROUP - 1, m->jnt_group[i]))])
                continue;

            // set data and name
            defSlider[0].pdata = d->qpos + m->jnt_qposadr[i];
            if (m->names[m->name_jntadr[i]])
                mju_strncpy(defSlider[0].name, m->names + m->name_jntadr[i],
                    mjMAXUINAME);
            else
                sprintf(defSlider[0].name, "joint %d", i);

            // set range
            if (m->jnt_limited[i])
                sprintf(defSlider[0].other, "%.4g %.4g",
                    m->jnt_range[2 * i], m->jnt_range[2 * i + 1]);
            else if (m->jnt_type[i] == mjJNT_SLIDE)
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
    for (i = 0; i < m->nu && itemcnt < mjMAXUIITEM; i++)
    {
        // skip if actuator group is disabled
        if (!vopt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP - 1, m->actuator_group[i]))])
            continue;

        // set data and name
        defSlider[0].pdata = d->ctrl + i;
        if (m->names[m->name_actuatoradr[i]])
            mju_strncpy(defSlider[0].name, m->names + m->name_actuatoradr[i],
                mjMAXUINAME);
        else
            sprintf(defSlider[0].name, "control %d", i);

        // set range
        if (m->actuator_ctrllimited[i])
            sprintf(defSlider[0].other, "%.4g %.4g",
                m->actuator_ctrlrange[2 * i], m->actuator_ctrlrange[2 * i + 1]);
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
    for (i = 0; i < NSECT0; i++)
    {
        oldstate0[i] = 0;
        if (ui0.nsect > i)
            oldstate0[i] = ui0.sect[i].state;
    }

    // get section open-close state, UI 1
    int oldstate1[NSECT1];
    for (i = 0; i < NSECT1; i++)
    {
        oldstate1[i] = 0;
        if (ui1.nsect > i)
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
    for (int i = 0; i < m->nq; i++)
    {
        sprintf(buf, i == m->nq - 1 ? "%g" : "%g ", d->qpos[i]);
        strcat(clipboard, buf);
    }
    strcat(clipboard, "'/>");

    // copy to clipboard
    glfwSetClipboardString(window, clipboard);
}



// millisecond timer, for MuJoCo built-in profiler
mjtNum timer(void)
{
    return (mjtNum)(1000 * glfwGetTime());
}



// clear all times
void cleartimers(void)
{
    for (int i = 0; i < mjNTIMER; i++)
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
    for (i = 0; i < mjNDISABLE; i++)
        settings.disable[i] = ((m->opt.disableflags & (1 << i)) != 0);
    for (i = 0; i < mjNENABLE; i++)
        settings.enable[i] = ((m->opt.enableflags & (1 << i)) != 0);

    // camera
    if (cam.type == mjCAMERA_FIXED)
        settings.camera = 2 + cam.fixedcamid;
    else if (cam.type == mjCAMERA_TRACKING)
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
    if (count > 0)
    {
        //mju_strncpy(filename, paths[0], 1000);
        //settings.loadrequest = 1;
        std::cout << "Drop file functionality disabled." << std::endl;
    }
}



// load mjb or xml model
int loadmodel(const char* filename)
{
    // make sure filename is not empty
    if (!filename[0]) {
        std::cout << "Model filename empty." << std::endl;
        return -1;
    }

    // check file existence 
    std::filesystem::path f{ filename };
    if (!std::filesystem::exists(f)) {
        std::cout << "Model file " << filename << " does not exist." << std::endl;
        return -2;
    }

    // load and compile
    char error[500] = "";
    mjModel* mnew = 0;
    if (strlen(filename) > 4 && !strcmp(filename + strlen(filename) - 4, ".mjb"))
    {
        mnew = mj_loadModel(filename, NULL);
        if (!mnew) {
            strcpy(error, "could not load binary model");
            return -3;
        }
    }
    else
        mnew = mj_loadXML(filename, NULL, error, 500);
    if (!mnew)
    {
        printf("%s\n", error);
        return -4;
    }

    // compiler warning: print and pause
    if (error[0])
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
    if (visuals_enabled) {
        mjv_makeScene(m, &scn, maxgeom);
        mjr_makeContext(m, &con, 50 * (settings.font + 1));
    }

    // clear perturbation state
    pert.active = 0;
    pert.select = 0;
    pert.skinselect = -1;

    if (visuals_enabled) {
        // align and scale view, update scene
        alignscale();
        mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

        // set window title to model name
        if (window && m->names)
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

    return 0;
}



//--------------------------------- UI hooks (for uitools.c) ----------------------------

// determine enable/disable item state given category
int uiPredicate(int category, void* userdata)
{
    switch (category)
    {
    case 2:                 // require model
        return (m != NULL);

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
    if ((state->dragrect == ui0.rectid) ||
        (state->dragrect == 0 && state->mouserect == ui0.rectid) ||
        state->type == mjEVENT_KEY)
    {
        // process UI event
        mjuiItem* it = mjui_event(&ui0, state, &con);

        // file section
        if (it && it->sectionid == SECT_FILE)
        {
            switch (it->itemid)
            {
            case 0:             // Save xml
                if (!mj_saveLastXML("mjmodel.xml", m, err, 200))
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
        else if (it && it->sectionid == SECT_OPTION)
        {
            switch (it->itemid)
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
                mjr_changeFont(50 * (settings.font + 1), &con);
                break;

            case 9:             // Full screen
                if (glfwGetWindowMonitor(window))
                {
                    // restore window from saved data
                    glfwSetWindowMonitor(window, NULL, windowpos[0], windowpos[1],
                        windowsize[0], windowsize[1], 0);
                }

                // currently windowed: switch to full screen
                else
                {
                    // save window data
                    glfwGetWindowPos(window, windowpos, windowpos + 1);
                    glfwGetWindowSize(window, windowsize, windowsize + 1);

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
        else if (it && it->sectionid == SECT_SIMULATION)
        {
            switch (it->itemid)
            {
            case 1:             // Reset
                if (m)
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
                mju_copy(d->qpos, m->key_qpos + i * m->nq, m->nq);
                mju_copy(d->qvel, m->key_qvel + i * m->nv, m->nv);
                mju_copy(d->act, m->key_act + i * m->na, m->na);
                mj_forward(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();
                break;

            case 7:             // Set key
                i = settings.key;
                m->key_time[i] = d->time;
                mju_copy(m->key_qpos + i * m->nq, d->qpos, m->nq);
                mju_copy(m->key_qvel + i * m->nv, d->qvel, m->nv);
                mju_copy(m->key_act + i * m->na, d->act, m->na);
                break;
            }
        }

        // physics section
        else if (it && it->sectionid == SECT_PHYSICS)
        {
            // update disable flags in mjOption
            m->opt.disableflags = 0;
            for (i = 0; i < mjNDISABLE; i++)
                if (settings.disable[i])
                    m->opt.disableflags |= (1 << i);

            // update enable flags in mjOption
            m->opt.enableflags = 0;
            for (i = 0; i < mjNENABLE; i++)
                if (settings.enable[i])
                    m->opt.enableflags |= (1 << i);
        }

        // rendering section
        else if (it && it->sectionid == SECT_RENDERING)
        {
            // set camera in mjvCamera
            if (settings.camera == 0)
                cam.type = mjCAMERA_FREE;
            else if (settings.camera == 1)
            {
                if (pert.select > 0)
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
        else if (it && it->sectionid == SECT_GROUP)
        {
            // remake joint section if joint group changed
            if (it->name[0] == 'J' && it->name[1] == 'o')
            {
                ui1.nsect = SECT_JOINT;
                makejoint(ui1.sect[SECT_JOINT].state);
                ui1.nsect = NSECT1;
                uiModify(window, &ui1, state, &con);
            }

            // remake control section if actuator group changed
            if (it->name[0] == 'A' && it->name[1] == 'c')
            {
                ui1.nsect = SECT_CONTROL;
                makecontrol(ui1.sect[SECT_CONTROL].state);
                ui1.nsect = NSECT1;
                uiModify(window, &ui1, state, &con);
            }
        }

        // stop if UI processed event
        if (it != NULL || (state->type == mjEVENT_KEY && state->key == 0))
            return;
    }

    // call UI 1 if event is directed to it
    if ((state->dragrect == ui1.rectid) ||
        (state->dragrect == 0 && state->mouserect == ui1.rectid) ||
        state->type == mjEVENT_KEY)
    {
        // process UI event
        mjuiItem* it = mjui_event(&ui1, state, &con);

        // control section
        if (it && it->sectionid == SECT_CONTROL)
        {
            // clear controls
            if (it->itemid == 0)
            {
                mju_zero(d->ctrl, m->nu);
                mjui_update(SECT_CONTROL, -1, &ui1, &uistate, &con);
            }
        }

        // stop if UI processed event
        if (it != NULL || (state->type == mjEVENT_KEY && state->key == 0))
            return;
    }

    // shortcut not handled by UI
    if (state->type == mjEVENT_KEY && state->key != 0)
    {
        switch (state->key)
        {
        case ' ':                   // Mode
            if (m)
            {
                settings.run = 1 - settings.run;
                pert.active = 0;
                mjui_update(-1, -1, &ui0, state, &con);
            }
            break;

        case mjKEY_RIGHT:
            // move one of the axes when the optimization is done
            if (m && n_x_final > 2) {
                // right
                x_final[2] += x_step[2];

                assume_posture_and_print_cost(false);
            }
            break;

        case mjKEY_LEFT:
            // move one of the axes when the optimization is done
            if (m && n_x_final > 2) {
                // left
                x_final[2] -= x_step[2];

                assume_posture_and_print_cost(false);
            }
            break;

        case mjKEY_DOWN:
            if (m && n_x_final > 1) {
                if (state->control) // back
                    x_final[0] -= x_step[0];
                else                // down
                    x_final[1] -= x_step[1];

                assume_posture_and_print_cost(false);
            }
            break;

        case mjKEY_UP:
            if (m && n_x_final > 1) {
                if (state->control)  // forward
                    x_final[0] += x_step[0];
                else                 // up
                    x_final[1] += x_step[1];

                assume_posture_and_print_cost(false);
            }
            break;

        // ROTATIONS
        case GLFW_KEY_KP_9:
            if (m && n_x_final > 3) {
                x_final[3] += x_step[3];

                assume_posture_and_print_cost(false);
            }
            break;

        case GLFW_KEY_KP_8:
            if (m && n_x_final > 3) {
                x_final[3] -= x_step[3];

                assume_posture_and_print_cost(false);
            }
            break;

        case GLFW_KEY_KP_6:
            if (m && n_x_final > 4) {
                x_final[4] += x_step[4];

                assume_posture_and_print_cost(false);
            }
            break;

        case GLFW_KEY_KP_5:
            if (m && n_x_final > 4) {
                x_final[4] -= x_step[4];

                assume_posture_and_print_cost(false);
            }
            break;

        case GLFW_KEY_KP_3:
            if (m && n_x_final > 5) {
                x_final[5] += x_step[5];

                assume_posture_and_print_cost(false);
            }
            break;

        case GLFW_KEY_KP_2:
            if (m && n_x_final > 5) {
                x_final[5] -= x_step[5];

                assume_posture_and_print_cost(false);
            }
            break;

        case GLFW_KEY_KP_ENTER:  // ctrl-numpad_enter
            if (m && n_x_final > 2 && state->control) {
                // save to file
                export_current_adjustment(x_final, n_x_final);
                if (verbose)
                    std::cout << "Exported current adjustment." << std::endl;
            }
            break;

        case mjKEY_PAGE_UP:         // select parent body
            if (m && pert.select > 0)
            {
                pert.select = m->body_parentid[pert.select];
                pert.skinselect = -1;

                // stop perturbation if world reached
                if (pert.select <= 0)
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
    if (state->type == mjEVENT_SCROLL && state->mouserect == 3 && m)
    {
        // emulate vertical mouse motion = 5% of window height
        mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * state->sy, &scn, &cam);

        return;
    }

    // 3D press
    if (state->type == mjEVENT_PRESS && state->mouserect == 3 && m)
    {
        // set perturbation
        int newperturb = 0;
        if (state->control && pert.select > 0)
        {
            // right: translate;  left: rotate
            if (state->right)
                newperturb = mjPERT_TRANSLATE;
            else if (state->left)
                newperturb = mjPERT_ROTATE;

            // perturbation onset: reset reference
            if (newperturb && !pert.active)
                mjv_initPerturb(m, d, &scn, &pert);
        }
        pert.active = newperturb;

        // handle double-click
        if (state->doubleclick)
        {
            // determine selection mode
            int selmode;
            if (state->button == mjBUTTON_LEFT)
                selmode = 1;
            else if (state->control)
                selmode = 3;
            else
                selmode = 2;

            // find geom and 3D click point, get corresponding body
            mjrRect r = state->rect[3];
            mjtNum selpnt[3];
            int selgeom, selskin;
            int selbody = mjv_select(m, d, &vopt,
                (mjtNum)r.width / (mjtNum)r.height,
                (mjtNum)(state->x - r.left) / (mjtNum)r.width,
                (mjtNum)(state->y - r.bottom) / (mjtNum)r.height,
                &scn, selpnt, &selgeom, &selskin);

            // set lookat point, start tracking is requested
            if (selmode == 2 || selmode == 3)
            {
                // copy selpnt if anything clicked
                if (selbody >= 0)
                    mju_copy3(cam.lookat, selpnt);

                // switch to tracking camera if dynamic body clicked
                if (selmode == 3 && selbody > 0)
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
                if (selbody >= 0)
                {
                    // record selection
                    pert.select = selbody;
                    pert.skinselect = selskin;

                    // compute localpos
                    mjtNum tmp[3];
                    mju_sub3(tmp, selpnt, d->xpos + 3 * pert.select);
                    mju_mulMatTVec(pert.localpos, d->xmat + 9 * pert.select, tmp, 3, 3);
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
    if (state->type == mjEVENT_RELEASE && state->dragrect == 3 && m)
    {
        // stop perturbation
        pert.active = 0;

        return;
    }

    // 3D move
    if (state->type == mjEVENT_MOVE && state->dragrect == 3 && m)
    {
        // determine action based on mouse button
        mjtMouse action;
        if (state->right)
            action = state->shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        else if (state->left)
            action = state->shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        else
            action = mjMOUSE_ZOOM;

        // move perturb or camera
        mjrRect r = state->rect[3];
        if (pert.active)
            mjv_movePerturb(m, d, action, state->dx / r.height, -state->dy / r.height,
                &scn, &pert);
        else
            mjv_moveCamera(m, action, state->dx / r.height, -state->dy / r.height,
                &scn, &cam);

        return;
    }
}

float calcFPS(std::vector<double> &ja_times) {
    return ((float) ja_times.size()) / (ja_times.back() - ja_times[0]);
}



//--------------------------- rendering and simulation ----------------------------------

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
    if (!m)
        return;

    // update scene
    mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

    // update watch
    if (settings.ui0 && ui0.sect[SECT_WATCH].state)
    {
        watch();
        mjui_update(SECT_WATCH, -1, &ui0, &uistate, &con);
    }

    // ipdate joint
    if (settings.ui1 && ui1.sect[SECT_JOINT].state)
        mjui_update(SECT_JOINT, -1, &ui1, &uistate, &con);

    // update info text
    if (settings.info)
        infotext(info_title, info_content, interval);

    // update profiler
    if (settings.profiler && settings.run)
        profilerupdate();

    // update sensor
    if (settings.sensor && settings.run)
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
    if (settings.profiler)
        smallrect.width = rect.width - rect.width / 4;

    // no model
    if (!m)
    {
        // blank screen
        mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

        // label
        if (settings.loadrequest)
            mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                "loading", NULL, &con);
        else
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect,
                "Drag-and-drop model file here", 0, &con);

        // render uis
        if (settings.ui0)
            mjui_render(&ui0, &uistate, &con);
        if (settings.ui1)
            mjui_render(&ui1, &uistate, &con);

        // finalize
        glfwSwapBuffers(window);

        return;
    }

    // render scene
    mjr_render(rect, &scn, &con);

    // show pause/loading label
    if (!settings.run || settings.loadrequest)
        mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
            settings.loadrequest ? "loading" : "pause", NULL, &con);

    // show ui 0
    if (settings.ui0)
        mjui_render(&ui0, &uistate, &con);

    // show ui 1
    if (settings.ui1)
        mjui_render(&ui1, &uistate, &con);

    // show help
    if (settings.help)
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content, &con);

    // show info
    if (settings.info)
        mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect,
            info_title, info_content, &con);

    // show profiler
    if (settings.profiler)
        profilershow(rect);

    // show sensor
    if (settings.sensor)
        sensorshow(smallrect);

    // finalize
    glfwSwapBuffers(window);
}


// NOT USED
// simulate in background thread (while rendering in main thread)
void simulate(void)
{
    // cpu-sim syncronization point
    double cpusync = 0;
    mjtNum simsync = 0;

    // run until asked to exit
    while (!settings.exitrequest)
    {
        // sleep for 1 ms or yield, to let main thread run
        //  yield results in busy wait - which has better timing but kills battery life
        if (settings.run && settings.busywait)
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // start exclusive access
        mtx.lock();

        // run only if model is present
        if (m)
        {
            // record start time
            double startwalltm = glfwGetTime();

            // running
            if (settings.run)
            {
                // record cpu time at start of iteration
                double tmstart = glfwGetTime();

                // out-of-sync (for any reason)
                if (d->time < simsync || tmstart<cpusync || cpusync == 0 ||
                    mju_abs((d->time - simsync) - (tmstart - cpusync))>syncmisalign)
                {
                    // re-sync
                    cpusync = tmstart;
                    simsync = d->time;

                    // clear old perturbations, apply new
                    mju_zero(d->xfrc_applied, 6 * m->nbody);
                    mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
                    mjv_applyPerturbForce(m, d, &pert);

                    // run single step, let next iteration deal with timing
                    mj_step(m, d);
                }

                // in-sync
                else
                {
                    // step while simtime lags behind cputime, and within safefactor
                    while ((d->time - simsync) < (glfwGetTime() - cpusync) &&
                        (glfwGetTime() - tmstart) < refreshfactor / vmode.refreshRate)
                    {
                        // clear old perturbations, apply new
                        mju_zero(d->xfrc_applied, 6 * m->nbody);
                        mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
                        mjv_applyPerturbForce(m, d, &pert);

                        // run mj_step
                        mjtNum prevtm = d->time;
                        mj_step(m, d);

                        // break on reset
                        if (d->time < prevtm)
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


//-------------------------------- init and deinit --------------------------------------

// initalize everything
void init(void)
{
    // print version, check compatibility
    if (verbose)
        printf("MuJoCo Pro version %.2lf\n", 0.01 * mj_version());
    if (mjVERSION_HEADER != mj_version())
        mju_error("Headers and library have different versions");

    // init GLFW, set timer callback (milliseconds)
    if (!glfwInit())
        mju_error("could not initialize GLFW");
    mjcb_time = timer;

    if (visuals_enabled) {
        // multisampling
        glfwWindowHint(GLFW_SAMPLES, 4);
        glfwWindowHint(GLFW_VISIBLE, 1);

        // get videomode and save
        vmode = *glfwGetVideoMode(glfwGetPrimaryMonitor());

        // create window
        window = glfwCreateWindow((2 * vmode.width) / 3, (2 * vmode.height) / 3,
            "Simulate", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            mju_error("could not create window");
        }

        // save window position and size
        glfwGetWindowPos(window, windowpos, windowpos + 1);
        glfwGetWindowSize(window, windowsize, windowsize + 1);

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
        settings.font = fontscale / 50 - 1;

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
}

// deinitialize everything
void deinit(void)
{
    // delete everything we allocated
    if (visuals_enabled)
        uiClearCallback(window);
    mj_deleteData(d);
    mj_deleteModel(m);
    if (visuals_enabled) {
        mjv_freeScene(&scn);
        mjr_freeContext(&con);
    }

    // deactive MuJoCo
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif
}


//---------------------------------------------------------------------------------------
//-------------------------------- /BASIC SIMULATE --------------------------------------
//---------------------------------------------------------------------------------------

//-------------------------------- gem and dof names to ids and back --------------------
std::vector<std::vector<int>> get_ps_geoms(
    std::string ps_prefix, int rows, int cols, 
    std::map<int, std::pair<int, int>>& geom_to_id)
{
    std::vector<std::vector<int>> answ(rows, std::vector<int>(cols, -1));
    char buf[mjMAXUINAME];
    std::string fmt(ps_prefix + "_%d_%d_geom");
    for (int i_geom = 0; i_geom < m->ngeom; i_geom++)
    {
        mju_strncpy(buf, m->names + m->name_geomadr[i_geom], mjMAXUINAME);
        // the second pair of arguments are about the compared string "buf"
        if (ps_prefix.compare(0, ps_prefix.size(), buf, 0, ps_prefix.size()))
            continue;
        // now to extract the row and column
        int row, col;
        if (sscanf_s(buf, fmt.c_str(), &row, &col) == 2) {
            answ[row][col] = i_geom;
            geom_to_id[i_geom] = std::make_pair(row, col);
        }
        else {
            if (verbose)
                std::cout << "Geom " << buf << " did not match row col scanning." << std::endl;
        }
    }

    // print a report
    int n_sensels = 0;
    for (auto row : answ)
        for (auto el : row)
            if (el >= 0)
                n_sensels++;
    if (verbose)
        std::cout << "Pressure sensor " << ps_prefix << " found " << n_sensels << " geoms" << std::endl;

    return answ;
}


std::vector<std::string> make_geom_to_name(void)
{
    std::vector<std::string> geom_to_name;
    char buf[mjMAXUINAME];
    for (int i_geom = 0; i_geom < m->ngeom; i_geom++)
    {
        mju_strncpy(buf, m->names + m->name_geomadr[i_geom], mjMAXUINAME);
        geom_to_name.push_back(buf);
    }
    return geom_to_name;
}


std::vector<int> get_geoms(
    const std::string str_to_match, 
    const std::vector<std::string> geom_to_name)
{
    std::vector<int> answ;
    std::regex e(str_to_match.c_str());
    std::cmatch cm;    // same as std::match_results<const char*> cm;
    for (int i_geom = 0; i_geom < geom_to_name.size(); i_geom++)
    {
        std::regex_match(geom_to_name[i_geom].c_str(), cm, e);
        if (cm.size() > 0)
            answ.push_back(i_geom);
    }
    return answ;
}


std::vector<int> get_hand_geoms(const std::vector<std::string> geom_to_name)
{
    std::vector<int> answ = get_geoms("^RA[3-9].*", geom_to_name);

    if (verbose) {
        std::cout << "Found " << answ.size() << " hand geoms:";
        for (const int& i_geom : answ)
            std::cout << " " << geom_to_name[i_geom];
        std::cout << std::endl;
    }
    return answ;
}


std::vector<int> get_thumb_geoms(const std::vector<std::string> geom_to_name)
{
    std::vector<int> answ = get_geoms("^RA[5-6][MPD]1_.*", geom_to_name);

    if (verbose) {
        std::cout << "Found " << answ.size() << " thumb geoms:";
        for (const int& i_geom : answ)
            std::cout << " " << geom_to_name[i_geom];
        std::cout << std::endl;
    }
    return answ;
}


std::vector<int> get_finger_geoms(const std::vector<std::string> geom_to_name)
{
    std::vector<int> answ = get_geoms("^RA[5-7][MPD][2-5]_.*", geom_to_name);

    if (verbose) {
        std::cout << "Found " << answ.size() << " finger geoms:";
        for (const int& i_geom : answ)
            std::cout << " " << geom_to_name[i_geom];
        std::cout << std::endl;
    }
    return answ;
}

std::vector<std::vector<int>> get_ps_dof_indices(
    const std::vector<std::string>& ps_prefixes, const std::vector<std::string>& axis_suffixes,
    const std::vector<int>& ja_indices, const std::vector<std::string>& ja_names,
    const std::vector<double>& joint_angles, std::vector<std::vector<double>>& initial_ja)
{
    // sure it can be rewritten to run faster through strncmp. Do that if you have nothing better to do
    std::vector<std::vector<int>> answ;
    initial_ja.clear();
    for (auto ps_prefix : ps_prefixes) {
        answ.push_back(std::vector<int>());
        initial_ja.push_back(std::vector<double>());
        for (size_t i_as = 0; i_as < axis_suffixes.size(); i_as++) {
            std::string tomatch(ps_prefix + ".*" + axis_suffixes[i_as]);
            std::regex e(tomatch);
            std::cmatch cm;
            for (size_t i_ja = 0; i_ja < ja_indices.size(); i_ja++)
            {
                std::regex_match(ja_names[i_ja].c_str(), cm, e);
                if (cm.size() > 0) {
                    answ.back().push_back(ja_indices[i_ja]);
                    initial_ja.back().push_back(joint_angles[i_ja]);
                    break;
                }
            }
            if (answ.back().size() == i_as) {
                throw std::out_of_range("Could not find joint corresponding to " + tomatch + ".");
            }
        }
    }

    return answ;
}

//-------------------------------- control model ----------------------------------------
int set_kinematics(
    const std::vector<int>& dof_indices,
    const std::vector<mjtNum>& dof_angles,
    const std::vector<mjtNum>& dof_vels,
    const std::vector<mjtNum>& dof_accs)
{
    // DEPRECATED
    // TODO future assert that lengths of all vectors are the same
    for (size_t i = 0; i < dof_indices.size(); i++)
    {
        d->qpos[dof_indices[i]] = dof_angles[i];
        d->qvel[dof_indices[i]] = dof_vels[i];
        d->qacc[dof_indices[i]] = dof_accs[i];
    }
    return 0;
}


int set_kinematics(
    const std::vector<int>& dof_indices, 
    const std::vector<mjtNum>& dof_angles)
{
    //std::vector<int>::iterator it;
    for (size_t i = 0; i < dof_indices.size(); i++)
    {
      if (dof_indices[i] >= 0) {
        d->qpos[dof_indices[i]] = dof_angles[i];

        // HACK FIX for pressure sensor
        // if (dof_indices[i] == 12 || dof_indices[i] == 13)
        //    d->qpos[dof_indices[i]] += 0.003;
        if (dof_indices[i] == 12) d->qpos[dof_indices[i]] *= -1;
      }
    }
    return 0;
}


std::vector<mjtNum> differential(const std::vector<mjtNum> left,
                                 const std::vector<mjtNum> right, mjtNum dt) {
  std::vector<mjtNum> diff;
  for (int i = 0; i < left.size(); i++)
    diff.push_back((right[i] - left[i]) / dt);
  return diff;
}


int set_all_kinematics(
    const int current_time_point,
    const std::vector<int>& dof_indices,
    const std::vector<std::vector<mjtNum>>& dof_angles_all,
    const std::vector<mjtNum>& time,
    std::vector<mjtNum>& accel) {

    // estimate dt
    mjtNum dt = (
        time[MIN(current_time_point + 1, time.size() - 1)] -
        time[MAX(current_time_point - 1, 0)]) / 2;

    // 3 points of position
    std::vector<mjtNum> pos = dof_angles_all[current_time_point];

    std::vector<mjtNum> pos_left =
        dof_angles_all[MAX(current_time_point - 1, 0)];
    std::vector<mjtNum> pos_right =
        dof_angles_all[MIN(current_time_point + 1, time.size() - 1)];

    // 2 points of velocity
    std::vector<mjtNum> vel_left = differential(pos_left, pos, dt);
    std::vector<mjtNum> vel_right = differential(pos, pos_right, dt);

    // average velocity from left and right estimate
    std::vector<mjtNum> vel;
    for (int i = 0; i < vel_left.size(); i++)
      vel.push_back((vel_left[i] + vel_right[i]) / 2);  

    // 1 point of acceleration
    std::vector<mjtNum> acc = differential(vel_left, vel_right, dt);

    // store to return
    accel.clear();
    for (size_t i = 0; i < dof_indices.size(); i++) {
      if (dof_indices[i] >= 0) {
        d->qpos[dof_indices[i]] = pos[i];
        d->qvel[dof_indices[i]] = vel[i];
        d->qacc[dof_indices[i]] = acc[i];

        // HACK FIX for pressure sensor - but do not care about the vel/acc for
        // it if (dof_indices[i] == 12 || dof_indices[i] == 13)
        //    d->qpos[dof_indices[i]] += 0.003;
        if (dof_indices[i] == 12) d->qpos[dof_indices[i]] *= -1;
      }
      accel.push_back(acc[i]);
    }
    
    return 0;
}


void add_acc(const std::vector<int>& dof_indices,
             const std::vector<mjtNum>& accel) {
  // for (size_t i = 0; i < dof_indices.size(); i++) {
  //  if (find(ignore_dof_indices.begin(), ignore_dof_indices.end(),
  //           dof_indices[i]) == ignore_dof_indices.end()) {
  //    d->qacc[dof_indices[i]] = accel[i] - d->qacc[dof_indices[i]];
  //  }
  //}
  // for (size_t i_v = 0; i_v < m->nv; i_v++) {
  //  d->qacc[i_v] = accel[i_v] - d->qacc[i_v];
  //}
  for (size_t i = 0; i < dof_indices.size(); i++) {
    if (dof_indices[i] >= 0) {
      d->qacc[dof_indices[i]] = accel[i] - d->qacc[dof_indices[i]];
    }
  }
}


//-------------------------------- display and colors -----------------------------------
void visualize_call(void)
{
    //--------- enter exclusive access
    mtx.lock();
    //std::cout << "inside the visualization mutex" << std::endl;

    // handle events (calls all callbacks)
    glfwPollEvents();
    // prepare to render
    prepare();

    //--------- exit exclusive access
    mtx.unlock();

    // render
    render(window);

    // don't overcall the mutex
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}


void visualize_loop(void)
{
    while (!glfwWindowShouldClose(window) && !settings.exitrequest) {
        visualize_call();
    }
}


void set_sensor_coloration(const std::vector<std::vector<mjtNum>>& ps_matrix, std::vector<std::vector<int>>& ps_geomids)
{
    //mjtNum totalAppliedForce = 0;
    for (int i_row = 0; i_row < ps_matrix.size(); i_row++)
        for (int i_col = 0; i_col < ps_matrix[i_row].size(); i_col++) {
            //if (ps_matrix[i_row][i_col] > 0)
            //    std::cout << i_row << " " << i_col << " " << ps_geomids[i_row][i_col] << " " << ps_geomids[43-i_row][i_col] << " " << ps_geomids[i_row][43-i_col] << " " << ps_geomids[43-i_row][43-i_col] << std::endl;
            if (ps_geomids[i_row][i_col] >= 0)
                if (ps_matrix[i_row][i_col] > 0) {
                    // set alpha to 1
                    m->geom_rgba[ps_geomids[i_row][i_col] * 4 + 3] = 1;
                    // m->geom_rgba[ps_geomids[i_row][i_col] * 4 + 0] = 1;  // make red
                    //totalAppliedForce += ps_matrix[i_row][i_col];
                }
                else
                    //set alpha to 0.2
                    m->geom_rgba[ps_geomids[i_row][i_col] * 4 + 3] = 0.4;
        }
    //mjtNum totalCurrentForce = 0;
    //for (int i_row = 0; i_row < ps_matrix.size(); i_row++)
    //    for (int i_col = 0; i_col < ps_matrix[i_row].size(); i_col++)
    //        totalCurrentForce += ps_matrix[i_row][i_col];
    //std::cout << "Total current force " << totalCurrentForce << " applied " << totalAppliedForce << 
    //    " difference " << totalCurrentForce - totalAppliedForce << "." << std::endl;
}


//-------------------------------- distances and contacts -------------------------------
struct MyContact_
{
    int ps_geomid = -1;
    mjtNum force = -1;
    int hand_geomid = -1;
    mjtNum dist = 0;
};
typedef struct MyContact_ MyContact;


std::vector<MyContact> match_sensors_hand(
    const std::vector<std::vector<mjtNum>>& ps_matrix, 
    const std::vector<std::vector<int>>& ps_geomids,
    const std::vector<int>& hand_geoms,
    mjtNum& unmatchedForce, mjtNum& matchedForce)  
{
    // TODO(future):
    // use  #include "engine/engine_collision_convex.h" function
    // // multi-point convex-convex collision, using libccd
    // int mjc_Convex(const mjModel * m, const mjData * d,
    //                mjContact * con, int g1, int g2, mjtNum margin);
    // Right now it is only included in the "src\engine\engine_collision_convex.c"
    // Since the source code is now open, when you feel like rebuilding MuJoCo, run it here instead of going through contacts.
    // It will increase performance by constraining the distance measurement only to the nonzero force sensels
    // right now the contact array is populated by all possible pairs

    // collect geom ids of all ps geoms with positive value, save their force values
    std::vector<MyContact> ps_hand_contacts;
    for (int i_row = 0; i_row < ps_matrix.size(); i_row++)
        for (int i_col = 0; i_col < ps_matrix[i_row].size(); i_col++) 
            if (ps_geomids[i_row][i_col] >= 0 && ps_matrix[i_row][i_col] > 0) {
                ps_hand_contacts.push_back(MyContact());
                ps_hand_contacts.back().ps_geomid = ps_geomids[i_row][i_col];
                ps_hand_contacts.back().force = ps_matrix[i_row][i_col];
            }
    
    // go through all contacts
    // find the shortest contact with the hand geoms, record the hand geom, and distance
    int handgeom;
    int i_mc;
    std::vector<int>::iterator it;
    for (int i_con = 0; i_con < d->ncon; i_con++)
    {
        handgeom = -1;
        i_mc = 0;
        for (; i_mc < ps_hand_contacts.size(); i_mc++) {
            if (ps_hand_contacts[i_mc].ps_geomid == d->contact[i_con].geom1 &&
                find(hand_geoms.begin(), hand_geoms.end(), d->contact[i_con].geom2) != hand_geoms.end()) {
                handgeom = d->contact[i_con].geom2;
                break;
            }
            else if (ps_hand_contacts[i_mc].ps_geomid == d->contact[i_con].geom2 &&
                find(hand_geoms.begin(), hand_geoms.end(), d->contact[i_con].geom1) != hand_geoms.end()) {
                handgeom = d->contact[i_con].geom1;
                break;
            }
        }
        if (handgeom < 0)
            continue;  // unrelated contact -- should not happen, since all contacts are prescribed in the model 

        if (ps_hand_contacts[i_mc].hand_geomid < 0 || d->contact[i_con].dist < ps_hand_contacts[i_mc].dist) {
            ps_hand_contacts[i_mc].dist = d->contact[i_con].dist;
            ps_hand_contacts[i_mc].hand_geomid = handgeom;
        }
    }
    // remove contacts that have not found a corresponding hand geom
    for (int i_mc = ps_hand_contacts.size() - 1; i_mc >= 0; i_mc--)
        if (ps_hand_contacts[i_mc].hand_geomid < 0) {
            unmatchedForce += ps_hand_contacts[i_mc].force;
            ps_hand_contacts.erase(ps_hand_contacts.begin() + i_mc);
        }
        else {
            matchedForce += ps_hand_contacts[i_mc].force;
        }

    // std::cout << "Found " << ps_hand_contacts.size() << " hand-sensor connections." << std::endl;
    return ps_hand_contacts;
}


void export_contacts(std::string& filename, std::vector<std::vector<MyContact>>& phc_storage,
    std::map<int, std::pair<int, int>>& geom_to_id, const std::vector<std::string>& geom_to_name)
{
    // directory has to exist
    std::ofstream f;
    f.open(filename);
    for (const std::vector<MyContact>& phc : phc_storage) {
        //std::cout << "Number of contacts: " << phc.size() << std::endl;
        for (const MyContact& c : phc) {
            f << geom_to_id[c.ps_geomid].first << "." << geom_to_id[c.ps_geomid].second << ":" << geom_to_name[c.hand_geomid] << ":" << c.dist << ",";
        }
        f << std::endl;
    }
    f.close();
}


//-------------------------------- externally applied forces ----------------------------
void apply_external_forces(const std::vector<MyContact>& ps_hand_contacts, const int direction=1) {
  int hand_body;
  // the largest face of the box
  const int axis = 2;
  mjtNum torque[3], force[3], n_world[3], face_center[3], R2[9];
  mjtNum *sz, *p, *body_pos;
  mju_zero(torque, 3);
  for (const MyContact& c : ps_hand_contacts) {
    hand_body = m->geom_bodyid[c.hand_geomid];

    // get the normal orientation of the geom
    mju_copy(R2, d->geom_xmat + 9 * c.ps_geomid, 9);
    R2[axis + 0] *= direction;
    R2[axis + 3] *= direction;
    R2[axis + 6] *= direction;
    n_world[0] = R2[axis + 0];
    n_world[1] = R2[axis + 3];
    n_world[2] = R2[axis + 6];

    // get the largest face center point
    sz = m->geom_size + 3 * c.ps_geomid;
    p = d->geom_xpos + 3 * c.ps_geomid;
    face_center[0] = p[0] + sz[axis] * n_world[0];
    face_center[1] = p[1] + sz[axis] * n_world[1];
    face_center[2] = p[2] + sz[axis] * n_world[2];

    body_pos = d->xpos + 3 * hand_body;

    // force equals normal vector times c.force
    mju_scl(force, n_world, c.force, 3);

    //std::cout << c.force << " " << force[0] << " " << force[1] << " " << force[2]
    //          << std::endl;

    // apply
    mj_applyFT(m, d, force, torque, face_center, hand_body, d->qfrc_applied);
  }
}

void add_external_forces_visualization(
    const std::vector<MyContact>& ps_hand_contacts,
    const int direction = 1) {
  int body;
  // the largest face of the box
  const int axis = 2;
  mjtNum torque[3], force[3], n_world[3], face_center[3], geomsize[3], R2[9];
  float geomrgba[4];
  mjtNum *sz, *p;
  mju_zero(torque, 3);
  for (const MyContact& c : ps_hand_contacts) {
    body = m->geom_bodyid[c.hand_geomid];

    // get the normal orientation of the geom
    mju_copy(R2, d->geom_xmat + 9 * c.ps_geomid, 9);
    R2[axis + 0] *= direction;
    R2[axis + 3] *= direction;
    R2[axis + 6] *= direction;
    n_world[0] = R2[axis + 0];
    n_world[1] = R2[axis + 3];
    n_world[2] = R2[axis + 6];

    // get the largest face center point
    sz = m->geom_size + 3 * c.ps_geomid;
    p = d->geom_xpos + 3 * c.ps_geomid;
    face_center[0] = p[0] + sz[axis] * n_world[0];
    face_center[1] = p[1] + sz[axis] * n_world[1];
    face_center[2] = p[2] + sz[axis] * n_world[2];

    // force equals normal vector times c.force
    mju_scl(force, n_world, c.force, 3);

    // VISUALIZE
    mjvGeom* g = scn.geoms + scn.ngeom;

    // size
    geomsize[0] = 3;   // linewidth
    geomsize[1] = 0;   // nothing?
    geomsize[2] = c.force;  // line length

    // rgba
    geomrgba[0] = 0.8f;
    geomrgba[1] = 0.0f;
    geomrgba[2] = 0.0f;
    geomrgba[3] = 0.8f;

    mjv_initGeom(g, mjGEOM_LINE, geomsize, face_center, R2, geomrgba);

    // add to geoms
    scn.ngeom++;
  }
}


std::vector<mjtNum> get_actuating_torques(
    const std::vector<int>& dof_indices) {
  std::vector<mjtNum> a(dof_indices.size(), 0.0);
  //mjtNum* tmp = new mjtNum(m->nv);
  //mj_mulJacTVec(m, d, tmp, d->xfrc_applied);  // external cartesian, should be 0s
  for (size_t i = 0; i < dof_indices.size(); i++) {
    if (dof_indices[i] >= 0)
      a[i] = d->qfrc_inverse[dof_indices[i]] - d->qfrc_applied[dof_indices[i]]/* - tmp[dof_indices[i]]*/;
  }
  return a;
}


//-------------------------------- arithmetics on contact -------------------------------
double dist_transform(const double dist, const double desired_min_dist=0.1, const double threshold_min_dist=0.0)
{
    // returns zero for dist between desired_min_dist and threshold_min_dist
    // diff from the edge otherwise
    if (dist > desired_min_dist)
        return fabs(dist - desired_min_dist);
    else if (dist < threshold_min_dist)
        return fabs(dist - threshold_min_dist);
    return 0.0;
}


double dist_transform2(const double dist, const double desired_min_dist = 0.01, const double threshold_min_dist = 0.0,
    const double offset=0.1)
{
    // returns square for dist between desired_min_dist and threshold_min_dist
    // diff from the edge otherwise
    if (dist > desired_min_dist)
        return fabs(dist - desired_min_dist) + offset;
    else if (dist < threshold_min_dist)
        return fabs(dist - threshold_min_dist) + offset;
    
    double halfwidth = (desired_min_dist - threshold_min_dist) / 2.;
    double answ = (dist - halfwidth) / halfwidth;

    return answ * answ * offset;
}


double dist_transform_square(const double dist, const double bend = 0.01)
{
    // lower than linear cost below `bend' and higher when higher
    double answ = fabs(dist) / bend;

    return answ * answ;
}


double average_distance(const std::vector<MyContact>& contacts)
{
    if (contacts.size() == 0)
        return HUGE_VAL;

    double answ = 0.0;
    for (const MyContact& v : contacts)
        answ += v.dist;
    return answ / contacts.size();
}


double weighted_average_distance(const std::vector<MyContact>& contacts)
{
    if (contacts.size() == 0)
        return HUGE_VAL;

    double answ = 0.0;
    for (const MyContact& v : contacts)
        answ += v.dist * fabs(v.force);
    return answ / contacts.size();
}


double weighted_average_transformed_distance(const std::vector<MyContact>& contacts)
{
    if (contacts.size() == 0)
        return HUGE_VAL;

    double total_force = 0.;
    double answ = 0.0;
    for (const MyContact& v : contacts) {
        total_force += fabs(v.force);
        answ += dist_transform_square(v.dist) * fabs(v.force);
    }
    return answ / contacts.size() /*/ total_force*/;
}


double median_distance(const std::vector<MyContact>& contacts)
{
    if (contacts.size() == 0)
        return HUGE_VAL;

    std::vector<double> dists;
    for (auto v : contacts)
        dists.push_back(v.dist);
    std::sort(dists.begin(), dists.end());
    if (dists.size() % 2)
        return dists[dists.size() / 2];
    else
        return dists[(dists.size() - 1) / 2];
}


//-------------------------------- optimization -----------------------------------------
// DEPRECATED and no longer needed/supported
// globals for optimization
struct ObjFunData_
{
    // some optimizer defaults
    int dim;
    double* x;
    double* lb;
    double* ub;

    // constant values
    std::vector < std::vector<int> > fitting_dof_indices;  // first index for left, second for right
    std::vector < std::vector<double> > fitting_initial_pos;  // first index for left, second for right
    std::vector < std::vector<int> >* le_ps_geoms;
    std::vector < std::vector<int> >* ri_ps_geoms;
    std::vector < std::vector<double> >* le_ps_matrix;
    std::vector < std::vector<double> >* ri_ps_matrix;
    std::vector <int>* hand_geoms;
    std::vector <int>* thumb_geoms;
    std::vector <int>* finger_geoms;

    // progress storage
    std::vector<double> fvals;
    std::vector<std::vector<double>> xs;
};
typedef struct ObjFunData_ ObjFunData;
ObjFunData* ofd_global = new ObjFunData;


double objective_function_test(unsigned n, const double* x, double* grad, void* my_func_data)
{
    double answ = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        answ += x[i] * x[i];
    }
    return answ;
}


void assume_optimized_posture_rel(const std::vector<double> x_vec, const ObjFunData* mfd)
{
    for (size_t i_fdi = 0; i_fdi < mfd->fitting_dof_indices.size(); i_fdi++)
        set_kinematics(mfd->fitting_dof_indices[i_fdi],
            MiscArrayFunctions::elementwise_sum(mfd->fitting_initial_pos[i_fdi], x_vec));

    // calculate contacts
    mj_forward(m, d);
}


std::vector<double> assume_optimized_posture_rel(const unsigned n, const double* x, const ObjFunData* mfd)
{
    std::vector<double> x_vec(x, x + n);
    assume_optimized_posture_rel(x_vec, mfd);

    return x_vec;
}


// on the current state
double cost_function(ObjFunData* mfd, bool lock_mutex=true)
{
    // -- lock thread for parallel computation on m, d
    if (lock_mutex)
        mtx.lock();
    
    mjtNum unmatchedForce = 0.;
    mjtNum matchedForce = 0.;
    // match sensors
    auto le_phs = match_sensors_hand(*(mfd->le_ps_matrix), *(mfd->le_ps_geoms), *(mfd->thumb_geoms), unmatchedForce, matchedForce);
    auto ri_phs = match_sensors_hand(*(mfd->ri_ps_matrix), *(mfd->ri_ps_geoms), *(mfd->finger_geoms), unmatchedForce, matchedForce);

    // -- unlock thread for parallel computation
    if (lock_mutex)
        mtx.unlock();

    // calculate matched distances
    double le_ad = weighted_average_transformed_distance(le_phs);
    double ri_ad = weighted_average_transformed_distance(ri_phs);

    if (!lock_mutex && verbose)
        std::cout << " " << le_ad << " " << ri_ad << " ";

    return le_ad + ri_ad;
}


std::mutex mtx_shared_data;

// does not run visual
double objective_function(unsigned n, const double* x, double* grad, void* my_func_data)
{
    ObjFunData* mfd = (ObjFunData*) my_func_data;

    // -- lock thread for parallel computation on m, d, and storing in mfd
    mtx.lock();

    // set kinematics
    std::vector<double> x_vec = assume_optimized_posture_rel(n, x, mfd);

    // -- unlock thread for parallel computation
    mtx.unlock();

    double answ = cost_function(mfd);

    // storage
    mtx_shared_data.lock();
    mfd->xs.push_back(std::vector<double>(x_vec));
    mfd->fvals.push_back(answ);

    // cmd output

    if (verbose)
        std::cout << "Eval " << mfd->fvals.size() << ": " << mfd->fvals.back() << " point";
    mtx_shared_data.unlock();

    if (verbose) {
        for (auto v : x_vec)
            std::cout << " " << v;
        std::cout << std::endl;
    }

    // pause evaluation while the simualtion is stopped
    while (true) {
        mtx.lock();
        if (settings.run) {
            mtx.unlock();
            break;
        }
        mtx.unlock();
        // don't overcall
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return answ;
}


void optimize_thread_function(void)
{
    // set up local optimizer
    nlopt_result nlopt_res;
    nlopt_opt opt_local = nlopt_create(NLOPT_LN_COBYLA, ofd_global->dim);
    nlopt_set_lower_bounds(opt_local, ofd_global->lb);
    nlopt_set_upper_bounds(opt_local, ofd_global->ub);
    nlopt_set_min_objective(opt_local, objective_function, (void*)ofd_global);
    nlopt_set_xtol_rel(opt_local, 1e-4);
    nlopt_set_initial_step(opt_local, x_step);

    // run the optimizer
    const double time_optimize_start = glfwGetTime();
    double minf = Basinhopping::optimize(
        opt_local, ofd_global->x, ofd_global->dim,
        ofd_global->lb, ofd_global->ub, (void*)ofd_global, 20);  // result
    //if ((nlopt_res = nlopt_optimize(opt_local, ofd_global->x, &minf)) < 0) {
    //    std::cout << "nlopt failed! Error code: " << nlopt_res << std::endl;
    //}
    //else {
    //    std::cout << "Found minimum with code " << nlopt_res << " at f(" << ofd_global->x[0];
    //    for (int i_x = 1; i_x < ofd_global->dim; i_x++)
    //        std::cout << ", " << ofd_global->x[i_x];
    //    std::cout << ") = " << minf << std::endl;
    //}

    double time_optimize = glfwGetTime() - time_optimize_start;

    if (verbose) {
        std::cout << "Spent " << time_optimize << " seconds to run eval " << ofd_global->fvals.size();
        std::cout << " times with average time " << time_optimize / ofd_global->fvals.size() << " s per eval." << std::endl;
    }

    // signal stopping
    mtx.lock();
    settings.done_optimizing = 1;
    mtx.unlock();
}


void assume_posture_and_print_cost(bool lock_mutex) {
    // -- lock thread for parallel computation on m, d, and storing in mfd
    if (lock_mutex)
        mtx.lock();

    // set kinematics
    std::vector<double> x_vec = assume_optimized_posture_rel(n_x_final, x_final, ofd_global);

    // -- unlock thread for parallel computation
    if (lock_mutex)
        mtx.unlock();

    if (verbose) {
        std::cout << "Current x:";
        for (int i_dim = 0; i_dim < n_x_final; i_dim++)
            std::cout << " " << x_final[i_dim];
        std::cout << ". Cost: " << cost_function(ofd_global, lock_mutex) << "." << std::endl;
    }
}


void export_current_adjustment(const std::vector<double> x_vec)
{
    if (adjustment_file.empty())
        return;
    IOFunctions::export_adjustment_file(
        adjustment_file,
        m, ofd_global->fitting_dof_indices, x_vec);
}


void export_current_adjustment(const double* x, const int n)
{
    if (n == 0)
        return;
    std::vector<double> x_vec(x, x + n_x_final);
    export_current_adjustment(x_vec);
}

bool CheckIfDirectoryExists(const std::string& path) {
    try {
        if (fs::exists(path) && fs::is_directory(path)) {
            return true; // Path exists and is a directory
        }
        else {
            return false; // Path doesn't exist or is not a directory
        }
    }
    catch (const std::exception& ex) {
        // An exception occurred while checking the path
        std::cerr << "Error checking path: " << ex.what() << std::endl;
        return false;
    }
}


//-------------------------------- main -------------------------------------------------
    

// run event loop
int main(int argc, const char** argv)
{
    //-------------------------- processing command line input
    std::string model_filename;
    // input joint angle filename
    std::string ja_file;
    // input pressure sensor filename
    std::string le_ps_file;
    std::string ri_ps_file;
    // output pressure sensor filename
    std::string le_ps_file_o;
    std::string ri_ps_file_o;
    // output torque filename
    std::string torque_file_o = "";
    // output torque filename
    std::string torque_nf_file_o = "";
    // optimization frame
    int optimization_frame = -1;
    // adjustment filename - for trial processing (as input) or optimization (as output)
    // std::string adjustment_file; NOW GLOBAL
    // help
    bool call_help = false;
    // video_filename to create video output dir in. If len>0, will write
    std::string video_filename = "";
    // skip to manual
    bool skip_to_manual = false;
    // no visualization
    // GLOBAL bool visuals_enabled = false;
    // don't save results to file
    bool skip_result_export = false;
    // do not move the thorax around
    bool vertical_thorax = true;
    // threshold for quality of the resulting matching (max portion unmatched)
    double quality_threshold = 0;
    // process inputs
    for (int i_arg = 1; i_arg < argc; ++i_arg) {
        // MuJoCo model filename.
        if (!strcmp(argv[i_arg], "-m") || !strcmp(argv[i_arg], "--model")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Model argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                model_filename = argv[++i_arg];
            }
        }
        // Joint angle CSV filename.
        else if (!strcmp(argv[i_arg], "--ja_in")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Input joint angles file argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                ja_file = argv[++i_arg];
            }
        }
        // writing video here
        else if (!strcmp(argv[i_arg], "--write_video")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Video output argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                video_filename = argv[++i_arg];
            }
        }
        // Left pressure sensor CSV filename.
        else if (!strcmp(argv[i_arg], "--leps_in")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Input left pressure sensor file argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                le_ps_file = argv[++i_arg];
            }
        }
        // Right pressure sensor CSV filename.
        else if (!strcmp(argv[i_arg], "--rips_in")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Input right pressure sensor file argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                ri_ps_file = argv[++i_arg];
            }
        }
        // the following few args are for running the whole trial
        // Left pressure sensor output CSV filename.
        else if (!strcmp(argv[i_arg], "--leps_ou")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Output left pressure sensor file argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                le_ps_file_o = argv[++i_arg];
            }
        }
        // Right pressure sensor CSV filename.
        else if (!strcmp(argv[i_arg], "--rips_ou")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Output right pressure sensor file argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                ri_ps_file_o = argv[++i_arg];
            }
        }
        // Torque output filename.
        else if (!strcmp(argv[i_arg], "--torque_ou")) {
          if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
            std::cout << "Output torque file argument "
                         "specified, but not provided. Aborting."
                      << std::endl;
            return -1;
          } else {
            torque_file_o = argv[++i_arg];
          }
        }
        // Torque computed without external forces filename.
        else if (!strcmp(argv[i_arg], "--torque_nf_ou")) {
          if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
            std::cout << "Output torque file argument "
                         "specified, but not provided. Aborting."
                      << std::endl;
            return -1;
          } else {
            torque_nf_file_o = argv[++i_arg];
          }
        }
        // quality threshold for the found matching - portion of force that can remain unmatched
        else if (!strcmp(argv[i_arg], "--quality_threshold")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Quality threshold argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                quality_threshold = atof(argv[++i_arg]);
            }
        }
        // the following few args are for running optimization on a frame
        // [--frame <frame_number>] to run optimization on.
        else if (!strcmp(argv[i_arg], "--frame")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Optimization frame argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                optimization_frame = atoi(argv[++i_arg]);
                printf("Optimization frame set to %d.\n", optimization_frame);
            }
        }
        // optimization file
        else if (!strcmp(argv[i_arg], "--adj")) {
            if (i_arg + 1 == argc || argv[i_arg + 1][0] == '-') {
                std::cout << "Adjustment file argument specified, but not provided. Aborting." << std::endl;
                return -1;
            }
            else {
                adjustment_file = argv[++i_arg];
            }
        }
        else if (!strcmp(argv[i_arg], "-h") || !strcmp(argv[i_arg], "--help"))
            call_help = true;
        else if (!strcmp(argv[i_arg], "--manual"))
            skip_to_manual = true;
        else if (!strcmp(argv[i_arg], "--no_visuals"))
            visuals_enabled = false;
        else if (!strcmp(argv[i_arg], "--skip_export"))
            skip_result_export = true;
        else if (!strcmp(argv[i_arg], "--vertical_thorax"))
            vertical_thorax = true;
        else if (!strcmp(argv[i_arg], "--verbose") || !strcmp(argv[i_arg], "-v"))
            verbose++;
        else {
            std::cout << "Unknown flag '" << argv[i_arg] << "'." << std::endl;
            call_help = true;
        }

        // HELP
        if (call_help) {
            std::cout << "This program is used to process trials to match pressure sensor sensels to specific digit segments or find an adjustment to sensor positions to better match the forces to the hand. Switch to optimization mode happens if the optional argument '--frame' is set. Accepted arguments (most are required):" << std::endl;
            std::cout << "  -m, --model <filename>	Path to MuJoCo model." << std::endl;
            std::cout << "  --ja_in <filename>		Path to the csv with joint angles of the trial." << std::endl;
            std::cout << "  --leps_in <filename>	Path to the csv with pressure sensor measurements of the trial from the left sensor. Should be aligned and synchronized with joint angles (sampled at the same time points)." << std::endl;
            std::cout << "  --rips_in <filename>	Path to the csv with pressure sensor measurements of the trial from the right sensor. Should be aligned and synchronized with joint angles (sampled at the same time points)." << std::endl;
            std::cout << "  --leps_ou <filename>	Where to save the matched between left sensor and hand segments. Needed if not in optimization mode." << std::endl;
            std::cout << "  --rips_ou <filename>	Where to save the matched between right sensor and hand segments. Needed if not in optimization mode." << std::endl;
            std::cout << "  --quality_threshold <float>	 If the unmatched force exceeds this portion of total force, the program will return -1." << std::endl;
            std::cout << "  --torque_ou <filename>  Filename for actuating torques."
                      << std::endl;
            std::cout << "  --torque_nf_ou <filename>  Filename for actuating torques computed without external forces."
                      << std::endl;
            std::cout << "  --no_visuals            Suppresses opening the window and visualization of the simualtion." << std::endl;
            std::cout << "  --skip_export           Skip exporint whatever results and data were generated." << std::endl;
            std::cout << "  --vertical_thorax       Makes the thorax vertical (does not change from the default model position)." << std::endl;
            std::cout << "  --verbose               Enable verbose reports." << std::endl;
            std::cout << "  -h, --help				Display this message." << std::endl;
            // todo add: starting with run
            return 0;
        }
    }
    if (verbose)
        std::cout << "This is simulate adjusted version 1.1.6." << std::endl;


    // check if all necessary files were provided
    if (model_filename.empty()) {
        std::cout << "Model filename not provided. Aborting." << std::endl;
        return -1;
    }
    else if (verbose)
        std::cout << "Model located at " << model_filename << " will be used." << std::endl;

    if (ja_file.empty()) {
        std::cout << "Input joint angle filename not provided. Aborting." << std::endl;
        return -1;
    }
    else if (verbose)
        std::cout << "Joint angles input file at " << ja_file << " will be used." << std::endl;
    if (le_ps_file.empty()) {
        std::cout << "Input left pressure sensor filename not provided. Aborting." << std::endl;
        return -1;
    }
    else if (verbose)
        std::cout << "Left pressure sensor input file at " << le_ps_file << " will be used." << std::endl;
    if (ri_ps_file.empty()) {
        std::cout << "Input right pressure sensor filename not provided. Aborting." << std::endl;
        return -1;
    }
    else if (verbose)
        std::cout << "Right pressure sensor input file at " << ri_ps_file << " will be used." << std::endl;

    if (verbose && quality_threshold > 0)
        printf("Quality threshold set to %f.\n", quality_threshold);

    // optimization_frame is used as indicator of the mode
    if (optimization_frame >= 0) {
        if (adjustment_file.empty()) {
            std::cout << "Optimization frame provided, but adjustment file was not. Aborting." << std::endl;
            return -1;
        }
        else if (verbose)
            std::cout << "Adjustment file at " << adjustment_file << " will be used." << std::endl;
        if (verbose)
            std::cout << "Starting in the OPTIMIZATION mode." << std::endl;
    }
    else {
        if (le_ps_file_o.empty()) {
            std::cout << "Output left pressure sensor filename not provided. Aborting." << std::endl;
            return -1;
        }
        else if (verbose)
            std::cout << "Left pressure sensor output file at " << le_ps_file_o << " will be used." << std::endl;
        if (ri_ps_file_o.empty()) {
            std::cout << "Output right pressure sensor filename not provided. Aborting." << std::endl;
            return -1;
        }
        else if (verbose)
            std::cout << "Right pressure sensor output file at " << ri_ps_file_o << " will be used." << std::endl;
        if (verbose)
            std::cout << "Starting in the MATCHING mode." << std::endl;
    }

    //-------------------------- initialize and load everything
    init();
    // TODO add to argin
    settings.run = 1;  // used to pause optimizations

    // get model filename
    if (verbose)
        std::cout << "Loading model " << model_filename << std::endl;
    loadmodel(model_filename.c_str());

    //-------------- load kinematic and kinetic data
    std::vector<int> ja_indices;
    std::vector<std::string> ja_names;
    std::vector<std::vector<mjtNum>> joint_angles;
    std::vector<mjtNum> ja_times;
    if (IOFunctions::import_csv_file(m, ja_file, ja_indices, ja_names, joint_angles, ja_times, verbose) < 0) {
        std::cout << "Error while loading joint angle CSV file " << ja_file << ", aborting." << std::endl;
        return -2;
    }

    std::vector<std::vector<std::vector<mjtNum>>> le_ps_matrix;
    std::vector<mjtNum> le_ps_times;
    if (IOFunctions::import_matrices(le_ps_file, le_ps_matrix, le_ps_times, verbose) < 0) {
        std::cout << "Error while loading left pressure sensor CSV file " << le_ps_file << ", aborting." << std::endl;
        return -3;
    }
    std::vector<std::vector<std::vector<mjtNum>>> ri_ps_matrix;
    std::vector<mjtNum> ri_ps_times;
    if (IOFunctions::import_matrices(ri_ps_file, ri_ps_matrix, ri_ps_times, verbose) < 0) {
        std::cout << "Error while loading right pressure sensor CSV file " << ri_ps_file << ", aborting." << std::endl;
        return -4;
    }
    // basic check that ps_time == ja_times
    if (ja_times.size() != le_ps_times.size() || ja_times.size() != ri_ps_times.size()) {
        std::cout << "Time series in joint angle and pressure sensor files do not match, aborting." << std::endl;
        return -5;
    }
    // load adjustment file
    std::vector<int> adjusted_dof_indices;
    std::vector<double> adjustments;
    if (!adjustment_file.empty()) {
        IOFunctions::import_adjustment_file(adjustment_file, ja_names, adjusted_dof_indices, adjustments);
        if (!adjustments.empty() && verbose) {
            std::cout << "Loaded adjustments for";
            for (auto ind : adjusted_dof_indices)
                std::cout << " " << ja_names[ind];
            std::cout << "." << std::endl << "\tValues:";
            for (auto adj : adjustments)
                std::cout << " " << adj;
            std::cout << "." << std::endl;
        }
    }

    // if want to stabilize thorax
    const std::vector<std::string> base_jas{"Thorax_rot1",
                                            "Thorax_rot2",
                                            "Thorax_rot3",
                                            "Thorax_tra1",
                                            "Thorax_tra2",
                                            "Thorax_tra3"};
    const std::vector<std::string> ignore_acc_jas{"Thorax_rot1",
                                            "Thorax_rot2",
                                            "Thorax_rot3",
                                            "Thorax_tra1",
                                            "Thorax_tra2",
                                            "Thorax_tra3",
                                            "ra_sternoclavicular_r2_d",
                                            "ra_sternoclavicular_r3_d",
                                            "ra_unrotscap_r3_d",
                                            "ra_unrotscap_r2_d",
                                            "ra_acromioclavicular_r2_d",
                                            "ra_acromioclavicular_r3_d",
                                            "ra_acromioclavicular_r1_d",
                                            "ra_unrothum_r1_d",
                                            "ra_unrothum_r3_d",
                                            "ra_unrothum_r2_d",
                                            "ra_shoulder1_r2_d",
                                            "ra_proximal_distal_r1_d",
                                            "ra_proximal_distal_r3_d"};
    std::vector<int> base_ja_ids;
    if (vertical_thorax)
        for (auto bja : base_jas) {
            base_ja_ids.push_back(mj_name2id(m, mjOBJ_JOINT, bja.c_str()));
            //std::cout << bja << " " << base_ja_ids.back() << std::endl;
        }
    std::vector<int> ignore_acc_ids;
    for (auto bja : ignore_acc_jas) {
      ignore_acc_ids.push_back(mj_name2id(m, mjOBJ_JOINT, bja.c_str()));
      // std::cout << bja << " " << base_ja_ids.back() << std::endl;
    }
    //const std::vector<std::string> other_ignore_jas{ "ps_halfwidth_tra", "ps_halfwidth_tra_d" };
    //for (auto oja : other_ignore_jas) {
    //    base_ja_ids.push_back(mj_name2id(m, mjOBJ_JOINT, oja.c_str()));
    //    //std::cout << bja << " " << base_ja_ids.back() << std::endl;
    //}
    //std::cout << "HERHEHRHEHREH" << base_ja_ids[0] << " " << base_ja_ids[1] << std::endl;
    
    // incorporate ignoring DOFs into the joint_indices
    if (vertical_thorax) {
      for (size_t i_dof = 0; i_dof < ja_indices.size(); i_dof++) {
        if (find(base_ja_ids.begin(), base_ja_ids.end(), ja_indices[i_dof]) !=
            base_ja_ids.end())
          ja_indices[i_dof] = -1;
      }
    }
    //// not used
    //std::vector<int> acc_ids; 
    //for (size_t i_dof = 0; i_dof < ja_indices.size(); i_dof++) {
    //  if (find(ignore_acc_ids.begin(), ignore_acc_ids.end(),
    //           ja_indices[i_dof]) == ignore_acc_ids.end())
    //    acc_ids.push_back(ja_indices[i_dof]);
    //  else
    //    acc_ids.push_back(-1);
    //}
    //std::cout << "ignored indices:";
    //for (const auto ji : base_ja_ids) std::cout << " " << ji;
    //std::cout << std::endl;
    //std::cout << "Joint Angle indices:";
    //for (const auto ji : ja_indices) std::cout << " " << ji;
    //std::cout << std::endl;
    //std::cout << "Acc indices:";
    //for (const auto ji : acc_ids) std::cout << " " << ji;
    //std::cout << std::endl;

    // if want to return a warning (>0) or quit properly, but with an error value(<0)
    int return_value = 0;

    //-------------- identify ids
    // TODO (future) fix hardcoded names of pressure sensors
    std::map<int, std::pair<int, int>> le_geom_to_id;
    std::map<int, std::pair<int, int>> ri_geom_to_id;
    auto le_ps_geoms = get_ps_geoms("LPS", le_ps_matrix[0].size(), le_ps_matrix[0][0].size(), le_geom_to_id);
    auto ri_ps_geoms = get_ps_geoms("RPS", ri_ps_matrix[0].size(), ri_ps_matrix[0][0].size(), ri_geom_to_id);

    // get hand geom ids
    std::vector<std::string> geom_to_name = make_geom_to_name();
    std::vector<int> hand_geoms = get_hand_geoms(geom_to_name);
    std::vector<int> thumb_geoms = get_thumb_geoms(geom_to_name);
    std::vector<int> finger_geoms = get_finger_geoms(geom_to_name);

    //-------------- preallocate
    // output
    std::vector<std::vector<MyContact>> ri_phc_storage;
    std::vector<std::vector<MyContact>> le_phc_storage;

    std::vector<std::vector<mjtNum>> actuating_torques_storage;
    std::vector<std::vector<mjtNum>> actuating_torques_nf_storage;

    // for quality control
    mjtNum unmatchedForce = 0.;
    mjtNum matchedForce = 0.;

    // NB does not from zero, zero is the time of contact
    std::vector<mjtNum> times(ja_times);
    std::vector<int> point_status(times.size(), 0);
    int i_current_point = 0;
    std::vector<mjtNum> accel;  // buffer for IK

    // track the time - for matching task
    const double time_start = glfwGetTime();
    double current_time = time_start;

    // Setup for video
    int frame_width, frame_height;
    std::shared_ptr<FrameSavingBuffer> frameSavingBuffer;
    GLubyte* frame_pixels = nullptr;
    if (video_filename.size()) {
        // Read the frame buffer
        glfwGetFramebufferSize(window, &frame_width, &frame_height);

        // Calculate frames per second from ja times
        float FPS = calcFPS(ja_times);
        if (verbose)
            std::cout << "FPS is: " << FPS << std::endl;

        frameSavingBuffer = std::make_shared<FrameSavingBuffer>(
            video_filename, FPS,
            frame_width,
            frame_height
            );
        frameSavingBuffer->log_timestamps = false;
        frame_pixels = new GLubyte[3 * (size_t)frame_width * (size_t)frame_height];
    }


    //-------------------------- main loop
    // apply adjustment file if it was loaded
    if (!adjusted_dof_indices.empty()) {
        for (size_t i_adof = 0; i_adof < adjusted_dof_indices.size(); i_adof++)
            for (size_t i_time = 0; i_time < times.size(); i_time++)
                joint_angles[i_time][adjusted_dof_indices[i_adof]] += adjustments[i_adof];
    }

    int disableflags = m->opt.disableflags;

    // the simulation will never go faster then realtime, and will pass through all timesteps
    while ((!visuals_enabled || !glfwWindowShouldClose(window)) && !settings.exitrequest && i_current_point < times.size())
    {
        bool write_frame = false;
        // get current time with 0 start of experiment
        current_time = glfwGetTime() - time_start;

        // go as fast as possible when noone is looking
        if (!visuals_enabled)
            current_time = mjMAXVAL;

        if (verbose)
            std::cout << "Current time: " << current_time << " s.\t\t\r";

        // identify next timepoint
        if (settings.run && (current_time >= (times[i_current_point] - times[0]))) {
            // enable contact etc
            m->opt.disableflags = disableflags;
            // process current timepoint
            // only pos is needed to compute the contacts and relative locations
            set_kinematics(ja_indices, joint_angles[i_current_point]);

            // set coloration of the sensor
            set_sensor_coloration(le_ps_matrix[i_current_point], le_ps_geoms);
            set_sensor_coloration(ri_ps_matrix[i_current_point], ri_ps_geoms);

            // compute stuff like contacts
            mj_forward(m, d);

            // get the distances
            le_phc_storage.push_back(match_sensors_hand(
                le_ps_matrix[i_current_point], le_ps_geoms, thumb_geoms, unmatchedForce, matchedForce));
            ri_phc_storage.push_back(match_sensors_hand(
                ri_ps_matrix[i_current_point], ri_ps_geoms, finger_geoms, unmatchedForce, matchedForce));

            // modify flags for torques
            m->opt.disableflags |=
                mjDSBL_EQUALITY | mjDSBL_LIMIT | mjDSBL_CONTACT;

            ////////// Torque 1
            // set the desired pos, vel, acc - needed for the inverse
            set_all_kinematics(i_current_point, ja_indices, joint_angles, times,
                               accel);
            // zero the forces
            mju_zero(d->qfrc_applied, m->nv);

            // refresh the cashes
            mj_forward(m, d);
            
            // fill in the accelerations
            set_all_kinematics(i_current_point, ja_indices, joint_angles, times,
                               accel);

            // apply the forces
            // thumb
            apply_external_forces(le_phc_storage.back(), -1);
            // fingers
            apply_external_forces(ri_phc_storage.back(), 1);

            // compute the torques
            mj_inverse(m, d);

            // save the torques
            actuating_torques_storage.push_back(
                get_actuating_torques(ja_indices));

            ////////// Torque 2 - without forces
            // set the desired pos, vel, acc - needed for the inverse
            set_all_kinematics(i_current_point, ja_indices, joint_angles, times,
                               accel);
            // remove the forces
            mju_zero(d->qfrc_applied, m->nv);

            // refresh the cashes
            mj_forward(m, d);

            // fill in the accelerations
            set_all_kinematics(i_current_point, ja_indices, joint_angles, times,
                               accel);

            // compute the torques
            mj_inverse(m, d);

            // save the torques
            actuating_torques_nf_storage.push_back(
                get_actuating_torques(ja_indices));

            // was this point successfully evaluated? - good foir debugging
            point_status[i_current_point] = 1;
            i_current_point++;

            write_frame = true;
        }

        // render
        if (visuals_enabled) {
            // handle events (calls all callbacks)
            glfwPollEvents();

            // prepare to render
            prepare();

            // thumb
            add_external_forces_visualization(le_phc_storage.back(), -1);
            // fingers
            add_external_forces_visualization(ri_phc_storage.back());

            // render
            render(window);
        }

        if (video_filename.size() && write_frame) {  //GL_RGB
            glReadPixels(0, 0, frame_width, frame_height, GL_RGB, GL_UNSIGNED_BYTE, frame_pixels);

            // Create an OpenCV Mat from the captured frame
            cv::Mat frame(frame_height, frame_width, CV_8UC3, frame_pixels);

            // Flip the frame vertically (if needed)
            cv::flip(frame, frame, 0);

            // OpenGl is RGB and OpenCV is BGR
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

            // Write the frame to the video
            frameSavingBuffer->add(Frame(
                frame.clone(),
                0,
                0,
                i_current_point
            ));

        }

    }

    // Release the video writer
    if (video_filename.size()) {
        frameSavingBuffer->stop_request = true;
        frameSavingBuffer->stop_wait();
        // Cleanup
        delete[] frame_pixels;
    }

    if (verbose)
        std::cout << std::endl;

    // export the data
    if (i_current_point == times.size() && !skip_result_export) {
        export_contacts(le_ps_file_o, le_phc_storage, le_geom_to_id, geom_to_name);
        if (verbose)
            std::cout << "Exported left pressure sensor contact matches to " << le_ps_file_o << std::endl;
        export_contacts(ri_ps_file_o, ri_phc_storage, ri_geom_to_id, geom_to_name);
        if (verbose)
            std::cout << "Exported right pressure sensor contact matches to " << ri_ps_file_o << std::endl;
        // torques
        if (!torque_file_o.empty()) {
            IOFunctions::export_timed_csv(torque_file_o, times, ja_names,
                                        actuating_torques_storage);
            if (verbose)
            std::cout << "Exported actuating torques to " << torque_file_o
                        << std::endl;
        }
        if (!torque_nf_file_o.empty()) {
          IOFunctions::export_timed_csv(torque_nf_file_o, times, ja_names,
                                        actuating_torques_nf_storage);
          if (verbose)
            std::cout << "Exported actuating no-force torques to " << torque_nf_file_o
                      << std::endl;
        }
    }

    if (quality_threshold > 0 && (unmatchedForce > (unmatchedForce + matchedForce) * quality_threshold)) {
        if (verbose)
            std::cout << "Unmatched force " << unmatchedForce << " N exceeded maximum portion (" << quality_threshold << ") of total trial force " << unmatchedForce + matchedForce << " N." << std::endl;
        return_value = -1;
    }
    if (verbose)
        std::cout << "Unmatched force " << unmatchedForce << " N, matched force " << matchedForce << " N, " << " total trial force " << unmatchedForce + matchedForce << " N." << std::endl;

    // done
    if (verbose)
        std::cout << "Deinitializing..." << std::endl;
    deinit();

    return return_value;
}
