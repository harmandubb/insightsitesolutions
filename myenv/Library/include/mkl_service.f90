!===============================================================================
! Copyright 2010-2022 Intel Corporation.
!
! This software and the related documents are Intel copyrighted  materials,  and
! your use of  them is  governed by the  express license  under which  they were
! provided to you (License).  Unless the License provides otherwise, you may not
! use, modify, copy, publish, distribute,  disclose or transmit this software or
! the related documents without Intel's prior written permission.
!
! This software and the related documents  are provided as  is,  with no express
! or implied  warranties,  other  than those  that are  expressly stated  in the
! License.
!===============================================================================

!  Content:
!   Intel(R) oneAPI Math Kernel Library (oneMKL) FORTRAN 95 interface for service routines
!*******************************************************************************

MODULE MKL_SERVICE

    IMPLICIT NONE

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DOMAIN_ALL  = 0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DOMAIN_BLAS = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DOMAIN_FFT  = 2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DOMAIN_VML  = 3
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DOMAIN_PARDISO = 4

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DYNAMIC_TRUE  = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_DYNAMIC_FALSE = 0

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_SSE4_2        = 0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX           = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX2          = 2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_MIC    = 3
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512        = 4
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_MIC_E1 = 5
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_E1     = 6
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_E2     = 7
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_E3     = 8
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_E4     = 9
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX2_E1       = 10
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_ENABLE_AVX512_E5     = 11

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_INTERFACE_LP64  = INT(Z"00000000")
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_INTERFACE_ILP64 = INT(Z"00000001")
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_INTERFACE_GNU   = INT(Z"00000002")
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_THREADING_INTEL = 0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_THREADING_SEQUENTIAL = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_THREADING_PGI = 2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_THREADING_GNU = 3
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_THREADING_TBB = 4

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_BRANCH         = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_ALL            = -1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_OFF            = 0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_UNSET_ALL      = 0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_BRANCH_OFF     = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AUTO           = 2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_COMPATIBLE     = 3
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_SSE2           = 4
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_SSSE3          = 6
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_SSE4_1         = 7
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_SSE4_2         = 8
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AVX            = 9
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AVX2           = 10
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AVX512_MIC     = 11
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AVX512         = 12
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AVX512_MIC_E1  = 13
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_AVX512_E1      = 14

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_STRICT         = 65536

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_SUCCESS                 =  0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_ERR_INVALID_SETTINGS    = -1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_ERR_INVALID_INPUT       = -2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_ERR_UNSUPPORTED_BRANCH  = -3
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_ERR_UNKNOWN_BRANCH      = -4
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_ERR_MODE_CHANGE_FAILURE = -8

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_PEAK_MEM_DISABLE        =  0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_PEAK_MEM_ENABLE         =  1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_PEAK_MEM_RESET          = -1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_PEAK_MEM                =  2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_MEM_MCDRAM              =  1

    INTERFACE
      SUBROUTINE MKL_GET_VERSION_STRING(BUF)
      CHARACTER*(*), INTENT(OUT) :: BUF
      END
    END INTERFACE

    INTERFACE
      DOUBLE PRECISION FUNCTION MKL_GET_CPU_FREQUENCY()
      END
    END INTERFACE

    INTERFACE
      DOUBLE PRECISION FUNCTION MKL_GET_MAX_CPU_FREQUENCY()
      END
    END INTERFACE

    INTERFACE
      DOUBLE PRECISION FUNCTION MKL_GET_CLOCKS_FREQUENCY()
      END
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_GET_CPU_CLOCKS(CPU_CLOCKS)
      INTEGER(8), INTENT(OUT) ::  CPU_CLOCKS
      END
    END INTERFACE

! Threading control functions

    INTERFACE
      INTEGER(4) FUNCTION MKL_GET_MAX_THREADS()
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_GET_NUM_STRIPES()
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_DOMAIN_GET_MAX_THREADS(DOMAIN)
      INTEGER(4), INTENT(IN) :: DOMAIN
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION  MKL_SET_NUM_THREADS_LOCAL(NTHRS)
      INTEGER(4), INTENT(IN) :: NTHRS
      END
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_SET_NUM_THREADS(NTHRS)
      INTEGER(4), INTENT(IN) :: NTHRS
      END
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_SET_NUM_STRIPES(NSTRP)
      INTEGER(4), INTENT(IN) :: NSTRP
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_DOMAIN_SET_NUM_THREADS(NTHRS,DOMAIN)
      INTEGER(4), INTENT(IN) :: NTHRS
      INTEGER(4), INTENT(IN) :: DOMAIN
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_GET_DYNAMIC()
      END
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_SET_DYNAMIC(MKL_DYNAMIC)
      INTEGER(4), INTENT(IN) :: MKL_DYNAMIC
      END
    END INTERFACE

! oneMKL Memory functions

    INTERFACE
      FUNCTION MKL_MALLOC(SIZE,ALIGN)
      USE ISO_C_BINDING
      INTEGER(KIND=C_INTPTR_T) MKL_MALLOC
      INTEGER(KIND=C_SIZE_T)   SIZE
      INTEGER(4) ALIGN
      END FUNCTION MKL_MALLOC
    END INTERFACE

    INTERFACE
      FUNCTION MKL_CALLOC(NUM,SIZE,ALIGN)
      USE ISO_C_BINDING
      INTEGER(KIND=C_INTPTR_T) MKL_CALLOC
      INTEGER(KIND=C_SIZE_T)   NUM,SIZE
      INTEGER(4) ALIGN
      END FUNCTION MKL_CALLOC
    END INTERFACE

    INTERFACE
      FUNCTION MKL_REALLOC(PTR,SIZE)
      USE ISO_C_BINDING
      INTEGER(KIND=C_INTPTR_T) MKL_REALLOC,PTR
      INTEGER(KIND=C_SIZE_T)   SIZE
      END FUNCTION MKL_REALLOC
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_FREE(PTR)
      USE ISO_C_BINDING
      INTEGER(KIND=C_INTPTR_T) PTR
      END SUBROUTINE MKL_FREE
    END INTERFACE

    INTERFACE
      INTEGER(8) FUNCTION MKL_MEM_STAT(N_BUFF)
      INTEGER(4), INTENT(OUT) :: N_BUFF
      END
    END INTERFACE

    INTERFACE
      INTEGER(8) FUNCTION MKL_PEAK_MEM_USAGE(RESET)
      INTEGER(4), INTENT(IN) :: RESET
      END
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_FREE_BUFFERS()
      END
    END INTERFACE

    INTERFACE
      SUBROUTINE MKL_THREAD_FREE_BUFFERS()
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_DISABLE_FAST_MM()
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_SET_MEMORY_LIMIT(MEM_TYPE,LIMIT)
      USE ISO_C_BINDING
      INTEGER(4), INTENT(IN) :: MEM_TYPE
      INTEGER(KIND=C_SIZE_T), INTENT(IN) :: LIMIT
      END
    END INTERFACE

! oneMKL Progress routine

    INTERFACE
      FUNCTION MKL_PROGRESS( THREAD, STEP, STAGE )
      INTEGER(4), INTENT(IN)    :: THREAD,STEP
      CHARACTER*(*), INTENT(IN) :: STAGE
      INTEGER          MKL_PROGRESS
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_ENABLE_INSTRUCTIONS(TYPE)
      INTEGER(4), INTENT(IN) :: TYPE
      END
    END INTERFACE

! oneMKL layer routines

    INTERFACE
      INTEGER(4) FUNCTION MKL_SET_INTERFACE_LAYER(MKL_INTERFACE)
      INTEGER(4), INTENT(IN) :: MKL_INTERFACE
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_SET_THREADING_LAYER(MKL_THREADING)
      INTEGER(4), INTENT(IN) :: MKL_THREADING
      END
    END INTERFACE

! oneMKL CBWR routines

    INTERFACE
      INTEGER(4) FUNCTION MKL_CBWR_GET(MKL_CBWR)
      INTEGER(4), INTENT(IN) :: MKL_CBWR
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_CBWR_SET(MKL_CBWR)
      INTEGER(4), INTENT(IN) :: MKL_CBWR
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_CBWR_GET_AUTO_BRANCH()
      END
    END INTERFACE

! oneMKL MPI routines

    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_BLACS_CUSTOM = 0
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_BLACS_MSMPI = 1
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_BLACS_INTELMPI = 2
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_BLACS_MPICH2 = 3

    INTERFACE
      INTEGER(4) FUNCTION MKL_SET_MPI(VENDOR, CUSTOM_LIBRARY_NAME)
      INTEGER(4), INTENT(IN) :: VENDOR
      CHARACTER*(*) :: CUSTOM_LIBRARY_NAME
      END
    END INTERFACE

! oneMKL verbose function

    INTERFACE
      INTEGER(4) FUNCTION MKL_VERBOSE(ENABLE)
      INTEGER(4), INTENT(IN) :: ENABLE
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_VERBOSE_OUTPUT_FILE(FILE_PATH)
      CHARACTER*(*), INTENT(IN) :: FILE_PATH
      END
    END INTERFACE

    INTERFACE
      INTEGER(4) FUNCTION MKL_SET_ENV_MODE(MODE)
      INTEGER(4), INTENT(IN) :: MODE
      END
    END INTERFACE

! Obsolete names and routines
    INTEGER (KIND=4), PARAMETER, PUBLIC :: MKL_CBWR_SSE3           = 5

!*******************************************************************************

END MODULE MKL_SERVICE
