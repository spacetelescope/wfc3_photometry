
c
c make sure it runs without GDC if it isn't needed... 
c

#define _PI_    3.141592653589793238462643d0

c
c need to make sure WFC3/IR subarrays are properly
c placed...


c
c enter the filenames by list
c

c
c the filename for PSFPERT doesn't actually get used; it's always
c treated as 'AUTO' 
c

c
c somehow the filter and PID and rdate are not getting populated
c  
c    jbts01ycq_flc 
c
c #   ARG000: ./hst2xym.2015.09.23.e
c #   ARG001: HMIN=7
c #   ARG002: FMIN=10000
c #   ARG003: PMAX=9e9
c #   ARG004: OUT=xymXYMpq
c
c--------------------------------------------------------------
c
c 2017.02.06
c * added capability to read _raw files for ACS
c
c 2016.06.02
c * need to add sky to possible output list...
c
c
c 2015.07.24
c * add option to do artificial stars...
c
c 2015.06.29
c * get it to read in the images and provide ALL of the  needed info 
c   at the time:  transformations, boundaries, etc.
c
c 2015.06.12
c * get it to reead in and work on wfc3ir images
c * I'll worry later about the geometric stuff
c
c 2015.05.06
c * turn the main finding part into a subroutine, 
c   so that I can put a wrapper around it... scary
c
c 2015.04.18
c * I turned off the normalization of the pertpsf... it was causing problems
c
c things yet to do
c * internalize the boundary thing (set based on chip at the beginning)
c * make sure I don't need the old-fashioned GCs anymore
c
c 2015.03.09c
c * put the new distortion effort into the measured positions
c * output the 2MASS stuff as region file
c
c 2015.03.06
c * compartmentalize/automate the way it does the distortion correction
c * simplifity the way it does the xy2rd mapping
c * simplifify the way it does the rd2uv mapping
c
c------------------------------------------------------------------------


c------------------------------------------------------------------------
c
c Monday Feb 23, 2015
c
c NEEDED IMPROVEMENTS
c 01) FLEXIBLE OUTPUT FILE uvMrdxyXYspqcnt
c 02) add imin/imax and jmin/jmax (make sure it works for satd, too!)  
c 03) output the important aspects right at the top (include detector)
c
c MAJOR RESTRUCTURING
c --> go thru args only once
c
c------------------------------------------------------------------------

c
c------------------------------------------------------------------------
c

#define _LINUX_  .true.     /* IS THE MACHINE BIG/LITTLE ENDIEN?       */
#define _NITPRT_      9     /* NUM OF ITERATIONS TO GET PRTPSF         */
#define _NPRTMX_      5     /* MAX NUMBER OF PERT ZONES                */
#define _FIELDs_     26     /* NUMBER OF OUTPUT FIELDS TO CHOOSE FROM  */
#define _NIMMXP_    130     /* MAX NUMBER OF IMAGES (FOR PERT)         */
#define _NIMMAX_  99999     /* MAX NUMBER OF IMAGES                    */
#define _OMAX_     9999     /* MAX NUMBER OF 2MASS STARS TO SEARCH FOR */
#define _NLMAX_   19999     /* MAX NUMBER OF STARS FOR PSF PERTURBN    */
#define _NSTMAX_ 999999     /* MAX NUMBER OF STARS IN AN IMAGE         */
#define _ARTMAX_ 999999     /* MAX NUMBER OF STARS IN AN IMAGE         */
#define _SKIPPX_ .false.
 
      program hst2xym
      implicit none

      integer TESTING123
 
      character*080 PROGNAME

      character*080 PSFFILE_000
      character*080 PSFFILE_INP
      character*080 PSFFILE_LIB
      character*080 PSFPERT_INP
      character*080 PSFFILE_PRT
      character*080 PSFFILE_USE
      character*200 FILENAME
      character*200 FILENAME_N(_NIMMAX_)
      character*080 WCSMODE
      character*080 CATMAT
      character*080 CATREG
      character*080 OUTDIR
      character*080 OUTDIRu

      character*080 ARTFILE_INP

      character*080 GDCFILE_000
      character*080 GDCFILE_INP
      character*080 GDCFILE_LIB
      character*080 GDCFILE_USE

      character*80 SHOW_ALL
      character*80 SHOW_USE
      character*80 SHOW_FND
      character*80 SHOW_REF
      character*80 SHOW_SUB
      character*80 SHOW_MSK

      character*80 STRING

      integer  i,  j
      integer iu
 
      integer HMIN
      real    PMAX
      real    FMIN

      integer iargc, NARG, NARGs
      integer LNC

      integer      ISIMG_NARG(999)
      character*80 STRNG_NARG(999)
 
      logical DOAPPHOT
      real    RAP
      integer SKI,SKO

      logical DOSATD 
      logical DOHSORT
      logical DOMSORT
      logical DOSTARDB
 
      integer HIFLAG
      common / HIFLAG_ / HIFLAG
      data HIFLAG / 70000 /
 
      integer LOFLAG
      common / LOFLAG_ / LOFLAG
      data LOFLAG / -0250 /
 
c----------------------------------
c
c here are the 9x9 fiducial PSFs
c
      integer NIM, NIMs
      real    QMAX
      real    CMIN, CMAX
      logical QCAND
 
      integer KSEL
 
      integer      NPERTs
      real         FOCUS_LEVELi
 
      real pertimg(501,1501)  ! can do up to 150 images...

      integer H, HHo, HHs
      character*249 HH(999)   ! max 999 header pages!

      integer       NLISTs
      character*80  OUTLIST_NL(9)  ! the suffix of the output file
      integer       NITEMSO_NL(9)  ! the number of elements to output

      integer       NREGs
      character*80  OUTLIST_NR(9)  ! the suffix of the output file
      integer       NITEMSO_NR(9)  ! the number of elements to output

      integer IMIN, IMAX
      integer JMIN, JMAX

c
c----------------------------------------------------------------
c   
      integer        N, Ns
      real*8         u_n(_NSTMAX_),  v_n(_NSTMAX_), mm_n(_NSTMAX_)
      real*8         x_n(_NSTMAX_),  y_n(_NSTMAX_),  m_n(_NSTMAX_)
      real*8        xx_n(_NSTMAX_), yy_n(_NSTMAX_)
      real*8        uu_n(_NSTMAX_), vv_n(_NSTMAX_)
      integer        k_n(_NSTMAX_)
      integer        h_n(_NSTMAX_)
      real          hh_n(_NSTMAX_)
      real*8         r_n(_NSTMAX_),  d_n(_NSTMAX_)
      integer        i_n(_NSTMAX_),  j_n(_NSTMAX_)
      real           p_n(_NSTMAX_)
      real           q_n(_NSTMAX_),  c_n(_NSTMAX_)
      real           s_n(_NSTMAX_)

      real*8 BDRY_XR(4,4), BDRY_YR(4,4)
      real*8 BDRY_XC(4,4), BDRY_YC(4,4)
      real*8 BDRY_UG(4,4), BDRY_VG(4,4)
      real*8 BDRY_RA(4,4), BDRY_DE(4,4)
      real*8 BDRY_UU(4,4), BDRY_VV(4,4)


c
c----------------------------------------------------------------
c   

      character*080 DIRECT
      character*080 PREFIX
      character*7   SUFFIX


      character*80 FILEOUT

      character*05 FILT_N(_NIMMAX_)
      real         EXPT_N(_NIMMAX_)
      real         RDAT_N(_NIMMAX_)
      integer      PROP_N(_NIMMAX_)
      integer      INST_N(_NIMMAX_)

      real*8  CRPIX1_USE, CRPIX2_USE
      real*8  CRVAL1_USE, CRVAL2_USE
      real*8   CD1_1_USE,  CD1_2_USE
      real*8   CD2_1_USE,  CD2_2_USE

      integer       NREG

      integer As
      real*8  xinp_a(_ARTMAX_)
      real*8  yinp_a(_ARTMAX_)
      real*8  minp_a(_ARTMAX_)

c
c----------------------------------------------------------------
c   

      do i = 001, 0501
      do j = 001, 1501
         pertimg(i,j) = 0.
         enddo
         enddo
 
      NARGs = iargc()

c
c if these aren't changed from their defaults; the routine will complain
c
      HMIN = -99
      FMIN = 999999

c-----------------------------------------------------------------------

      GDCFILE_000 = 'AUTO'
      GDCFILE_INP = 'AUTO'
      GDCFILE_LIB = 'NONE'
      GDCFILE_USE = 'NONE'

      PSFFILE_000 = 'AUTO'
      PSFFILE_INP = 'NONE'
      PSFFILE_LIB = 'NONE'
      PSFFILE_USE = 'NONE'
      PSFPERT_INP = 'NONE'
      PSFFILE_PRT = 'NONE'

      DOSATD     = .true.
      DOHSORT    = .false.
      DOMSORT    = .false.
      DOSTARDB   = .false.

      NPERTs       = 0
      FOCUS_LEVELi = 0.

      As = 0

      KSEL       =  0
      QMAX       =  9.99
      CMIN       = -1.00
      CMAX       =  0.10
      QCAND      =  .false.

      SHOW_ALL   = 'NO'
      SHOW_USE   = 'NO'
      SHOW_FND   = 'NO'
      SHOW_REF   = 'NO'
      SHOW_MSK   = 'X'
      SHOW_SUB   = 'NO'

      WCSMODE = 'FIND'
      CATMAT  = 'NONE'
      CATREG  = 'NONE'
      NLISTs   = 0  
      NREGs    = 0  
 
      IMIN = -999
      IMAX = 9999
      JMIN = -999
      JMAX = 9999

      OUTDIR = 'NONE'

      DOAPPHOT =  .false.

      PMAX = 999999

      if (iargc().eq.0) then
         print*,'                                          '
         print*,'This routine takes sevaral args (3 req)   '
         print*,'                                          '
         print*,'hst2xym HMIN=5                            '
         print*,'        FMIN=1000                         '
         print*,'        [PSF=AUTO/FILE/APPHOT]            '
         print*,'        [GDC=AUTO/FILE/NONE]  (distortn)  '
         print*,'        [PMAX=99999]                      '
         print*,'        [QMAX= 0.1]                       '
         print*,'        [CMIN=-1.0]                       '
         print*,'        [CMIN= 0.1]                       '
         print*,'        [SHOW_USE=NO/AUTO/X] image search '
         print*,'        [SHOW_FND=NO/AUTO/X] find report  '
         print*,'        [SHOW_REF=NO/AUTO/X] ref-frame img'
         print*,'        [SHOW_SUB=NO/AUTO/X] subt image   '
         print*,'        [SHOW_MSK=NO/AUTO/X] mask         '
         print*,'        [SHOW_ALL=NO/AUTO/X] show all     '
         print*,'        [PERTn=YES/AUTO/FILE]             '
         print*,'        [FOCUS=-1(find),0(mid),rF]        '
         print*,'        [REF-FRAME PARAMETERS]            '
         print*,'        [IMIN=500] [IMAX=510]             '
         print*,'        [JMIN=922] [JMAX=932]             '
         print*,'        [WCS=FIND/1ST]                    '
         print*,'           [HDR=]                         '
         print*,'           [RDC=]                         '
         print*,'        [MSORT+] [HSORT+]                 '
         print*,'        [STARDB+]                         '
         print*,'        [OUT=uvMrd]                       '
         print*,'        [REG=xy,XY,uv,rd]                 '
         print*,'        [CATMAT=2MASS/1st]                '
         print*,'        [CATREG=FILE]                     '
         print*,'        [CAT_UV=FILE.uv...]               '
         print*,'        [CAT_RD=FILE.rd...]               '
         print*,'        [ARTXYM=FILE.xym]                 '
         print*,'        [FITSs=LIST OF FILES, ONE / LINE] '
         print*,'        IMG1.fits IMG2.fits ...           '
         print*,'                                          '
         print*,'"PSF=APPHOT 2.99 6 9"                     '
         print*,'      ---> for aperture photy             '
         print*,'           w/ap (r<2.99) and              '
         print*,'           sky bet 6 & 9                  '
         print*,'                                          '
         print*,'KSEL  -- use only this chip number        ' 
         print*,'QMAX  -- this is the maximum allow QFIT   ' 
         print*,'CMIN  -- this is the minimum central exess'
         print*,'         relative to a point source       '
         print*,'         (use to throw out resolved obj)  '
         print*,'CMAX  -- this is the maximum central exess'
         print*,'         relative to a point source       '
         print*,'         (use to throw out CRs)           '
         print*,'QCAND -- requires both q and c together,  '
         print*,'         (not or)                         '
         print*,'                                                   '
         print*,'REFERENCE FRAME PARAMETERS...                      '
         print*,'     OPTION#0:  AUTO (default)                     '
         print*,'     OPTION#1:  "WCS=IMAGE_WCS.txt"                '
         print*,'                 (wcs info in textfile)            '
         print*,'     OPTION#2:  "HDR=IMAGE.fits"                   '
         print*,'                 (wcs info from fits header)       '
         print*,'     OPTION#3:  "RDC=(00:23:13.5,+33:44:13.22)"    '
         print*,'                 (reference center for 50mas frame)'
         print*,'                 (hard format)                     '
         print*,'                                                   '
         print*,'OUT= output file can be made flexibly              '
         print*,'     u,v = ref frame u and v position              '
         print*,'     m   = instrumental mag                        '
         print*,'     M   = instrumental mag (1000s, pix-area corr) '
         print*,'     V   = VEGA MAG (not available yet)            '
         print*,'     x,y = raw chip-dependent x y coord            '
         print*,'     k   = chip number (neg for xy in abtd system) '
         print*,'     X,Y = dist corr x y                           ' 
         print*,'     h   = isolation index (pixel based)           '
         print*,'     H   = isolation index (output-file based)     '
         print*,'     r,R = Right Ascension (R in hex)              '
         print*,'     r,D = Declination (D in hex)                  '
         print*,'     s   = sky value                               '
         print*,'     p   = brightest pixel value                   '
         print*,'     q   = quality of fit (0 = perfect)            '
         print*,'     c   = chisq of fit                            '
         print*,'     k,K = chip number (cap gives K1)              '
         print*,'                                                   '
         print*,'OUTDIR= specify directory for OUT and REG files... '
         print*,'       OUTDIR=NONE  -- default, put in run dir     '
         print*,'       OUTDIR=IMAGE -- put in image directories    '
         print*,'       OUTDIR=<DIR> -- send to specific directory  '
         print*,'                                                   '
         stop
         endif

      do H = 1, 999
         write(HH(H),'(''#'',248('' ''))')
         enddo 
 
      call getarg(0,PROGNAME)
 
      write(HH(1),'(''#'')')
      write(HH(2),'(''#--------------------------------------------'')')
      write(HH(3),'(''# ARGUMENTS '')')
      write(HH(4),'(''#--------------------------------------------'')')
      write(HH(5),'(''#   ARG'',i4.4,'': '',a80)') 0,PROGNAME 
      HHo = 5

c
c----------------------------------------------------------------
c



      print*,' '
      print*,' '
      write(*,'(''ARG'',i4.4,2x,80a)') 0,PROGNAME
      NIMs = 0 
      do NARG = 1, NARGs
         ISIMG_NARG(NARG) = 0
         call getarg(NARG,FILENAME)
         HHo = HHo + 1
         write(HH(HHo),'(''#   ARG'',i4.4,'': '',a80)') NARG,FILENAME
         STRNG_NARG(NARG) = FILENAME
         write(*,'(''ARG'',i4.4,2x,80a)') NARG,FILENAME
         if (FILENAME(1:5).eq.'HMIN=') then
            read(FILENAME(6:80),*) HMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'FMIN=') then
            read(FILENAME(6:80),*) FMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'PMAX=') then
            read(FILENAME(6:80),*) PMAX
            goto 555
            endif
         if (FILENAME(1:4).eq.'PSF=') then
            PSFFILE_000 = FILENAME(5:80)
            PSFFILE_INP = FILENAME(5:80)
            goto 555
            endif
         if (FILENAME(1:4).eq.'GDC=') then
            GDCFILE_000 = FILENAME(5:80)
            goto 555
            endif
         if (FILENAME(1:4).eq.'NOHI') then
            HIFLAG = 999999
            goto 555
            endif
         if (FILENAME(1:4).eq.'WCS=') then
            WCSMODE = FILENAME(5:80)
            goto 555
            endif
         if (FILENAME(1:4).eq.'HDR=') then
            WCSMODE = FILENAME
            goto 555
            endif
         if (FILENAME(1:4).eq.'RDC=') then
            WCSMODE = FILENAME
            goto 555
            endif
         if (FILENAME(1:7).eq.'DOSATD+') then
            DOSATD = .true.
            goto 555
            endif
         if (FILENAME(1:7).eq.'DOSATD-') then
            DOSATD = .false.
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_USE=') then
            SHOW_USE = FILENAME(10:80)
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_FND=') then
            SHOW_FND = FILENAME(10:80)
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_REF=') then
            SHOW_REF=FILENAME(10:80)
            if (SHOW_REF(1:2).ne.'NO'.and.
     .          SHOW_REF(1:4).ne.'AUTO') then
                print*,' SHOW_REF: ',SHOW_REF
                print*,' (can only be NO or AUTO)'
                stop
                endif
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_SUB=') then
            SHOW_SUB=FILENAME(10:80)
            if (SHOW_SUB(1:2).ne.'NO'.and.
     .          SHOW_SUB(1:4).ne.'AUTO') then
                print*,' SHOW_SUB: ',SHOW_SUB
                print*,' (can only be NO or AUTO)'
                stop
                endif
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_MSK=') then
            SHOW_MSK=FILENAME(10:80)
            if (SHOW_MSK(1:1).ne.'X' .and.
     .          SHOW_MSK(1:2).ne.'NO'.and.
     .          SHOW_MSK(1:4).ne.'AUTO') then
                print*,' SHOW_MSK: ',SHOW_MSK
                print*,' (can only be X, NO or AUTO)'
                stop
                endif
            goto 555
            endif
         if (FILENAME(1:6).eq.'FOCUS=') then
            read(FILENAME(7:80),*) FOCUS_LEVELi
            print*,'---> FOCUS_LEVELi: ',FOCUS_LEVELi
            goto 555
            endif 
         if (FILENAME(1:4).eq.'PERT') then
            if (FILENAME(5:5).lt.'0'.or.FILENAME(5:5).gt.'9') then
               print*,' '
               print*,'PERTn MUST HAVE AN n VALUE BETWEEN 0 AND 9'
               print*,'ARG: ',FILENAME(1:20)
               print*,' '
               stop 
               endif
            read(FILENAME(5:5),*) NPERTs
            print*,'        ---> NPERTs: ',NPERTs
            if (FILENAME(6:6).ne.'=') then
               print*,' '
               print*,'The 6th character of PERT must be = '
               print*,' '
               stop
               endif
            if (NPERTs.gt._NPRTMX_) then
               print*,'hst2xym --- '
               print*,'        --->  NPERTs : ',NPERTs
               print*,'        ---> _NPRTMX_: ',_NPRTMX_
               stop
               endif
            PSFPERT_INP = FILENAME(7:80) 
            goto 555
            endif
         if (FILENAME(1:5).eq.'KSEL=') then
            read(FILENAME(6:20),*) KSEL
            goto 555
            endif
         if (FILENAME(1:5).eq.'QMAX=') then
            read(FILENAME(6:20),*) QMAX
            goto 555
            endif
         if (FILENAME(1:5).eq.'CMIN=') then
            read(FILENAME(6:20),*) CMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'CMAX=') then
            read(FILENAME(6:20),*) CMAX
            goto 555
            endif
         if (FILENAME(1:5).eq.'QCAND') then
            if (FILENAME(6:6).ne.'+'.and.
     .          FILENAME(6:6).ne.'-') then
                print*,'QCAND flag must be + or -...'
                stop 
                endif
            if (FILENAME(6:6).eq.'+') QCAND = .true.
            if (FILENAME(6:6).eq.'-') QCAND = .false.
            print*,'---> QCAND: ',QCAND 
            goto 555
            endif
         if (FILENAME(1:5).eq.'IMIN=') then
            read(FILENAME(6:80),*) IMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'IMAX=') then
            read(FILENAME(6:80),*) IMAX
            goto 555
            endif
         if (FILENAME(1:5).eq.'JMIN=') then
            read(FILENAME(6:80),*) JMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'JMAX=') then
            read(FILENAME(6:80),*) JMAX
            goto 555
            endif
         if (FILENAME(1:6).eq.'MSORT+') then
            DOMSORT = .true.
            goto 555
            endif
         if (FILENAME(1:6).eq.'HSORT+') then
            DOHSORT = .true.
            goto 555
            endif
         if (FILENAME(1:7).eq.'STARDB+') then
            DOSTARDB = .true.
            goto 555
            endif
         if (FILENAME(1:7).eq.'OUTDIR=') then
            OUTDIR = FILENAME(8:80)
            print*,'                   '
            print*,'  Specify OUTDIR: '
            print*,'     OUTDIR = ',OUTDIR
            print*,'                   '
            goto 555
            endif 
         if (FILENAME(1:4).eq.'OUT=') then
            NLISTs = NLISTs + 1
            if (NLISTs.gt.9) stop 'NLISTs.gt.9'
            OUTLIST_NL(NLISTs) = FILENAME(5:80)
            NITEMSO_NL(NLISTs) = 0
            do i = 1, 80 
               if (OUTLIST_NL(NLISTs)(i:i).ne.' ') NITEMSO_NL(NLISTs)= i
               enddo
            print*,'       --->    NLISTs = ',NLISTs
            print*,'       --->    NITEMs = ',NITEMSO_NL(NLISTs)
            print*,'       --->   OUTLIST = ',
     .                        OUTLIST_NL(NLISTs)(1:NITEMSO_NL(NLISTs))
            goto 555
            endif 
         if (FILENAME(1:4).eq.'REG=') then
            NREGs = NREGs + 1
            if (NREGs.gt.9) stop 'NREGs.gt.9'
            OUTLIST_NR(NREGs) = FILENAME(5:80)
            NITEMSO_NR(NREGs) = 0
            do i = 1, 80 
               if (OUTLIST_NR(NREGs)(i:i).ne.' ') NITEMSO_NR(NREGs)= i
               enddo
            print*,'       --->     NREGs = ',NREGs
            print*,'       --->    NITEMs = ',NITEMSO_NR(NREGs)
            print*,'       --->   OUTLIST = ',
     .                     OUTLIST_NR(NREGs)(1:NITEMSO_NR(NREGs))
            if (NITEMSO_NR(NREGs).ne.2) 
     .          stop 'can only have two positions in reg file'
            if (OUTLIST_NR(NREGs)(1:2).ne.'xy'.and.
     .          OUTLIST_NR(NREGs)(1:2).ne.'XY'.and.
     .          OUTLIST_NR(NREGs)(1:2).ne.'uv'.and.
     .          OUTLIST_NR(NREGs)(1:2).ne.'rd') 
     .              stop 'CAN ONLY MAKE REGION OF xy XY uv RD'
            goto 555
            endif 
         if (FILENAME(1:7).eq.'CATMAT=') then
            CATMAT = FILENAME(8:80)
            goto 555
            endif
         if (FILENAME(1:7).eq.'CATREG=') then
            CATREG = FILENAME(8:80)
            goto 555
            endif
         if (FILENAME(1:7).eq.'ARTXYM=') then
            ARTFILE_INP = FILENAME(8:80)
            open(19,file=ARTFILE_INP,status='old')
            As = 0
    1       read(19,'(a80)',end=2) STRING
            if (STRING(1:1).eq.'#') goto 1
            As = As + 1
            read(STRING,*) xinp_a(As), yinp_a(As), minp_a(As)
            goto 1
    2       continue
            print*,'READ IN --- As: ',As
            close(19)
            goto 555
            endif
         if (FILENAME(1:6).eq.'FITSs=') then
            print*,'OPEN: ',FILENAME(7:80)
            open(19,file=FILENAME(7:80),status='old')
  888       read(19,'(a80)',end=889) STRING
            if (STRING(1:1).eq.'#') goto 888
            iu = 0
            do i = 1, 75
               if (STRING(i:i+4).eq.'.fits') iu = i+4
               enddo
            if (iu.eq.0) stop 'FITSs FILENAME IN LIST MUST HAVE .fits'
            NIMs = NIMs + 1
            if (NIMs.gt._NIMMAX_) then
                print*,'     NIMs : ',NIMs
                print*,'  _NIMMAX_: ',_NIMMAX_
                stop 'NIMs.gt._NIMMAX_'
                endif
            FILENAME_N(NIMs) = STRING(1:iu+4)
            write(*,'(''        ---> NIM'',i4.4,1x,80a)') 
     .                               NIMs,FILENAME_N(NIMs)
            goto 888
  889       continue
            print*,'---> NIMs: ',NIMs
            goto 555
            endif

c
c at this point, we should only have FITS images left...
c
         iu = 0
         do i = 1, 75
            if (FILENAME(i:i+4).eq.'.fits') iu = i
            enddo
         if (iu.eq.0) stop 'UNRECOGNIZED PARAMETER'

         NIMs = NIMs + 1
         HHo  = HHo  - 1
         if (NIMs.gt._NIMMAX_) stop 'NIMs.gt._NIMMAX_'
         FILENAME_N(NIMs) = FILENAME(1:LNC(FILENAME,200)) 
         write(*,'(''        ---> NIM'',i4.4,1x,80a)') NIMs,FILENAME
  555    continue
         enddo



c----------------------------------------------------------
c
c everything has been read in now from the command line, 
c check to make sure that we have enough information to do 
c what needs to be done...
c
      if (HMIN.eq.-99) then
         print*,'--> COMMAND LINE MUST INCLUDE HMIN'
         stop
         endif
      if (FMIN.eq.-99) then
         print*,'--> COMMAND LINE MUST INCLUDE FMIN'
         stop
         endif
      if (PSFFILE_INP(1:4).eq.'NULL') then
         print*,'--> COMMAND LINE MUST INCLUDE PSF'
         stop
         endif
      if (NIMs.eq.0) then
         print*,'--> COMMAND LINE MUST INCLUDE AT LEAST ONE IMAGE' 
         stop
         endif

      print*,' '
      print*,' '
      write(*,'(''            '')')
      write(*,'(''            '')')
      write(*,'(''   OUTPUT FROM PROGRAM hst2xym  '')')
      write(*,'(''            '')')
      write(*,'(''            '',a80)')  PROGNAME
      write(*,'(''            '')')
      write(*,'(''     HMIN: '',i3  )')  HMIN
      write(*,'(''     FMIN: '',f19.1)') FMIN
      write(*,'(''     PMAX: '',f19.1)') PMAX
      write(*,'('' PSFFILE0: '',a80)')   PSFFILE_000
      write(*,'('' GDCFILE0: '',a80)')   GDCFILE_000
      write(*,'('' PSFFILEI: '',a80)')   PSFFILE_INP
      write(*,'('' GDCFILEI: '',a80)')   GDCFILE_INP
      write(*,'('' DOAPPHOT: '',l1)')    DOAPPHOT
      write(*,'(''          '')')
      write(*,'(''   DOSATD: '',l01)') DOSATD
      write(*,'(''  GDC_INP: '',80a)') GDCFILE_INP
      write(*,'(''  GDC_LIB: '',80a)') GDCFILE_LIB
      write(*,'(''  GDC_USE: '',80a)') GDCFILE_USE
      write(*,'(''  PSF_INP: '',80a)') PSFFILE_INP
      write(*,'(''  PSF_LIB: '',80a)') PSFFILE_LIB
      write(*,'('' PERT_INP: '',80a)') PSFPERT_INP
      write(*,'(''   NPERTs: '',i1)')  NPERTs
      write(*,'(''  WCSMODE: '',80a)') WCSMODE
      write(*,'(''     KSEL: '',i2  )') KSEL
      write(*,'(''     QMAX: '',f6.3)') QMAX
      write(*,'(''     CMIN: '',f6.3)') CMIN
      write(*,'(''     CMAX: '',f6.3)') CMAX
      write(*,'(''    QCAND: '',l1)')   QCAND
      write(*,'(''           '')')  
      write(*,'(''     NIMs: '',i5)') NIMs
      do NIM = 1, NIMs
         print*,'NIM: ',NIM,FILENAME_N(NIM)(1:60)
         enddo
      write(*,'(''           '')')  

      if (PSFFILE_INP(1:6).eq.'APPHOT') then
          write(*,'(''#PSFFILEI: '',80a)') PSFFILE_INP
          read(PSFFILE_INP(7:80),*) RAP,SKI,SKO
          write(*,'(''# ---> APERTURE RAP: '',f8.4)') RAP
          write(*,'(''# --->    INNER SKI: '',i3)') SKI
          write(*,'(''# --->    OUTER SKO: '',i3)') SKO
          DOAPPHOT = .true.
          endif

      do NIM = 1, NIMs
         GDCFILE_INP = GDCFILE_000
         PSFFILE_INP = PSFFILE_000
          
         call sub_hst2xym(HH,HHo,HHs,
     .                    HMIN,FMIN,PMAX,
     .                    HIFLAG,WCSMODE,DOSATD,
     .                    SHOW_ALL,SHOW_USE,SHOW_FND,
     .                    SHOW_REF,SHOW_SUB,SHOW_MSK,
     .                    NPERTs,
     .                    KSEL,QMAX,CMIN,CMAX,QCAND,
     .                    IMIN,IMAX,JMIN,JMAX,DOMSORT,DOHSORT,
     .                    CATMAT, CATREG, 
     .                    FILENAME_N(NIM),NIM,NIMs,
     .                       PSFFILE_INP,
     .                       GDCFILE_INP,
     .                    Ns,  u_n,  v_n, mm_n,
     .                         x_n,  y_n,  m_n,  k_n,
     .                        xx_n, yy_n, 
     .                        uu_n, vv_n,
     .                         h_n, hh_n,
     .                         r_n,  d_n,
     .                         i_n,  j_n,  p_n,
     .                         q_n,  c_n,  s_n,
     .                    BDRY_XR, BDRY_YR,
     .                    BDRY_XC, BDRY_YC,
     .                    BDRY_UG, BDRY_VG,
     .                    BDRY_RA, BDRY_DE,
     .                    BDRY_UU, BDRY_VV,
     .                    FILT_N(NIM),EXPT_N(NIM),
     .                    RDAT_N(NIM),PROP_N(NIM),INST_N(NIM),
     .                    CRPIX1_USE, CRPIX2_USE,
     .                    CRVAL1_USE, CRVAL2_USE,
     .                    CD1_1_USE,  CD1_2_USE,
     .                    CD2_1_USE,  CD2_2_USE,
     .                    2,                   ! VERBOSE
     .                    As,xinp_a,yinp_a,minp_a,
     .                    1,FOCUS_LEVELi,DOSTARDB) 


         call dirstrip(FILENAME_N(NIM),DIRECT,PREFIX,SUFFIX)

         OUTDIRu = OUTDIR
         if (OUTDIR(1:4).eq.'NONE' ) OUTDIRu = './'
         if (OUTDIR(1:5).eq.'IMAGE') OUTDIRu = DIRECT

         if (NLISTs.ne.0) 
     .   call output_lists(NLISTs,NITEMSO_NL,OUTLIST_NL,
     .                     OUTDIRu,PREFIX,
     .                     HH, HHs,
     .                     Ns,  u_n,  v_n, mm_n,
     .                          x_n,  y_n,  m_n,  k_n,
     .                         xx_n, yy_n, 
     .                         uu_n, vv_n,
     .                          h_n, hh_n,
     .                          r_n,  d_n,
     .                          i_n,  j_n,  p_n,
     .                          q_n,  c_n,  s_n)


         if (NREGs.ne.0) 
     .   call output_regs(NREGs,NITEMSO_NR,OUTLIST_NR,
     .                    OUTDIRu,PREFIX,HIFLAG,
     .                    HH, HHs,
     .                    Ns,  u_n,  v_n, mm_n,
     .                         x_n,  y_n,  m_n,  k_n,
     .                        xx_n, yy_n, 
     .                        uu_n, vv_n,
     .                         h_n, hh_n,
     .                         r_n,  d_n,
     .                         i_n,  j_n,  p_n,
     .                         q_n,  c_n)



         enddo ! NIM = 1, NIMs


      stop
      end 




 
#include "/user/jayander/FORTRAN/ROUTINES/GEN/oiojor.f"
#include "/user/jayander/FORTRAN/ROUTINES/IMAGE/fnd_hloc_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/IMAGE/fnd_ploc_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/SORT/rbubble.f"
#include "/user/jayander/FORTRAN/ROUTINES/STAT/histmode_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/STRING/dirstrip.f"
#include "/user/jayander/FORTRAN/ROUTINES/GEN/rclip.f"
#include "/user/jayander/FORTRAN/ROUTINES/GEN/dclip.f"
#include "/user/jayander/FORTRAN/ROUTINES/PHOT/mbar_sky_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/mbar_skyopt_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/PHOT/apphot_NAXIS.f"


#include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdr.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i2r.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i4r.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4e.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_r4.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_b1.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_i4.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4_3D.f"

#include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/rpsf_phot.f"
#include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/zero_fix.f"
#include "/user/jayander/FORTRAN/ROUTINES/BARSIG/rbarsigs.f"

#include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/max_contig_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/peak_sat_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/sat_phot_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/max_contig_drz.f"
#include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/peak_sat_drz.f"

#include "/user/jayander/FORTRAN/WCS/rd2x.f"
#include "/user/jayander/FORTRAN/WCS/rd2y.f"
#include "/user/jayander/FORTRAN/WCS/xy2r.f"
#include "/user/jayander/FORTRAN/WCS/xy2d.f"
#include "/user/jayander/FORTRAN/WCS/rd2lb_G.f"
#include "/user/jayander/FORTRAN/WCS/lb2rd_G.f"
#include "/user/jayander/FORTRAN/WCS/rd2lb_E.f"
#include "/user/jayander/FORTRAN/WCS/lb2rd_E.f"
#include "/user/jayander/FORTRAN/WCS/string2ra.f"
#include "/user/jayander/FORTRAN/WCS/string2de.f"
#include "/user/jayander/FORTRAN/WCS/ra2sexig.f"
#include "/user/jayander/FORTRAN/WCS/de2sexig.f"
#include "/user/jayander/FORTRAN/WCS/fill_wcs_info.f"
#include "/user/jayander/FORTRAN/WCS/inside_poly.f"
#include "/user/jayander/FORTRAN/WCS/find_obj.f"

#include "/user/jayander/FORTRAN/PSFs/PERT/find_psfpert.f"
#include "/user/jayander/FORTRAN/PSFs/PERT/output_psfperts.f"
#include "/user/jayander/FORTRAN/PSFs/PERT/ras2pertpsf.f"
#include "/user/jayander/FORTRAN/PSFs/PERT/rbarsigsw.f"
#include "/user/jayander/FORTRAN/PSFs/PERT/rpsfpert_phot.f"

#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/infofits_stdpsf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/loadfits_stdpsf_flexibl.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/locpsfij_stdpsf_preload.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/stdpsf2psfbar.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/writfits_stdpsf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/savefits_stdpsf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/writfits_psfpert2std.f"

#include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/infofits_stdpbf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/locpsfijf_stdpbf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/findfocus_stdpbf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/loadstdpsf_stdpbf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/loadfits_stdpbf.f"
#include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/rpsf_phot_ij_stdpsf.f"

#include "/user/jayander/FORTRAN/ROUTINES/IMAGE/pix2mask.f"

#include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/find_xyzXX_NAXIS.f"
#include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/z_xyoptXX_NAXIS.f"

#include "/user/jayander/FORTRAN/FITSIO/PUTCHAR/img2putchar_r4.f"
#include "/user/jayander/FORTRAN/ROUTINES/SORT/quiksort.f"

#include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre_i4.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre_r4.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre_r8.f"

#include "/user/jayander/FORTRAN/2MASS/sub_query_2mass.f"

#include "/user/jayander/FORTRAN/WCS/sub_find_ifilt_acswfc.f"
#include "/user/jayander/FORTRAN/WCS/sub_find_ifilt_wfc3uv.f"
#include "/user/jayander/FORTRAN/2MASS/find_offset_2MASS.f"
#include "/user/jayander/FORTRAN/WCS/find_ifilt_ACSWFC.f"
#include "/user/jayander/FORTRAN/WCS/find_ifilt_WFC3UV.f"

#include "/user/jayander/FORTRAN/GC/STDGDCs/extract_stdgc.f"
#include "/user/jayander/FORTRAN/GC/STDGDCs/xryr2mc_stdgc.f"
#include "/user/jayander/FORTRAN/GC/STDGDCs/xryr2xcyc_stdgc.f"
#include "/user/jayander/FORTRAN/GC/STDGDCs/xcyc2xryr_stdgc.f"

#include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_r4_WCSLTV.f"

#include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/sub_hst2xym_info.f"
#include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/fill_charout.f"
#include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/find_rdate.f"  
#include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/lnc.f"
#include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/output_lists.f"
#include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/output_regs.f"

#include "/user/jayander/FORTRAN/FITSIO/WFC3IR/read_wfc3ir_flt_IMPROVED.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i2e.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i4e.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4_4D.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/rdate_header.f"

#include "/user/jayander/FORTRAN/FITSIO/WFC3IR/read_wfc3ir_flt_full.f"
#include "/user/jayander/FORTRAN/FITSIO/ACSWFC/read_acswfc_flt_full.f"
#include "/user/jayander/FORTRAN/FITSIO/ACSHRC/read_acshrc_flt_full.f"
#include "/user/jayander/FORTRAN/FITSIO/WFC3UV/read_wfc3uv_flt_full.f"
#include "/user/jayander/FORTRAN/FITSIO/WFPC2/read_wfpc2_flt_full.f"
#include "/user/jayander/FORTRAN/FITSIO/HSTDRZ/read_hstdrz_full.f"
#include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_j2r.f"

#include "/user/jayander/FORTRAN/ROUTINES/STRING/str_contains.f"
#include "/user/jayander/FORTRAN/ROUTINES/NOISE/noisesig.f"

#include "/user/jayander/FORTRAN/PSFs/PERT/smoo_psfpert_new.f"

#include "/user/jayander/FORTRAN/ROUTINES/SATN/fitsat_ccdXthenY.f"
#include "/user/jayander/FORTRAN/ROUTINES/SATN/rbarsigw_empir.f"

