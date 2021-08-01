#!/bin/bash

MATCHINGRESULTS="GlobalMuonTracks.root matching.log MatchingPlane_eV*.png ML_Evaluation*.root"
CHECKRESULTS="GlobalMuonChecks.root checks.log"

Usage()
{
    echo "USAGE"
    exit
}

updatecode() {
  cp -r ${SCRIPTDIR}/generators/* ${SCRIPTDIR}/*.bin ${SCRIPTDIR}/*.xml ${SCRIPTDIR}/include ${SCRIPTDIR}/*.C ${SCRIPTDIR}/*.h ${SCRIPTDIR}/*.cxx ${SCRIPTDIR}/*.py ${SCRIPTDIR}/macrohelpers ${OUTDIR}
}

generateMCHTracks()
{

  mkdir -p ${OUTDIR}
  updatecode
  pushd ${OUTDIR}

  #sed -i -e s/NPIONS/${NPIONS}/g Config.C
  #sed -i -e s/NMUONS/${NMUONS}/g Config.C

  echo "Generating MCH tracks on `pwd` ..."
  #echo ${MCHGENERATOR}_${NPIONS}pi_${NMUONS}mu_${NEV_}evts  > GENCFG

  ## 1) aliroot generation of MCH Tracks
  export SEED=${SEED:-"123456"}
  echo ${NEV_} > nMCHEvents
  export NEV=${NEV_}
  rm -rf MatcherGenConfig.txt
  #bash ./runtest.sh -n ${NEV_} | tee aliroot_MCHgen.log

  ## 2) aliroot conversion of MCH tracks to temporary format
  echo " Converting MCH Tracks to O2-compatible format"
  aliroot -e 'gSystem->Load("libpythia6_4_25")' -b -q -l "ConvertMCHESDTracks.C+(\".\")" | tee MCH-O2Conversion.log
  popd
  echo " Finished MCH track generation `realpath ${OUTDIR}`"

}


generateMFTTracks()
{

  if ! [ -f "${OUTDIR}/Kinematics.root" ]; then
    echo " ERROR! MCH Tracks Kinematics.root not found on `realpath ${OUTDIR}/Kinematics.root` ... exiting."
    exit
  fi

  if ! [ -z ${UPDATECODE+x} ]; then updatecode ; fi
  pushd ${OUTDIR}

  echo "Generating MFT Tracks `pwd` ..."

  NEV_=`cat nMCHEvents`
  ## O2 simulation and generation of MFT tracks using same Kinematics.root
  o2-sim -g extkin --extKinFile Kinematics.root -m PIPE ITS MFT ABS SHIL -e TGeant3 -n ${NEV_} -j $JOBS | tee O2Sim.log
  o2-sim-digitizer-workflow ${CUSTOM_SHM} -b --skipDet TPC,ITS,TOF,FT0,EMC,HMP,ZDC,TRD,MCH,MID,FDD,PHS,FV0,CPV >  O2Digitizer.log
  o2-mft-reco-workflow ${CUSTOM_SHM} -b > O2Reco.log
  popd
  echo " Finished MFT Track generation on `realpath ${OUTDIR}`"

}

runMatching()
{

  if ! [ -f "${OUTDIR}/tempMCHTracks.root" ]; then
    echo " Nothing to Match... MCH Tracks not found on `realpath ${OUTDIR}/tempMCHTracks.root` ..."
    EXITERROR="1"
  fi

  if ! [ -f "${OUTDIR}/mfttracks.root" ]; then
    echo " Nothing to Match... MFT Tracks not found on `realpath ${OUTDIR}/mfttracks.root` ..."
    EXITERROR="1"
  fi

  if ! [ -z ${EXITERROR+x} ]; then exit ; fi

  if [ -d "${OUTDIR}" ]; then
    if ! [ -z ${UPDATECODE+x} ]; then updatecode ; fi

    pushd ${OUTDIR}
    echo "Matching MCH & MFT Tracks on `pwd` ..."
    ## MFT MCH track matching & global muon track fitting:
    root -e "gSystem->Load(\"libO2MCHTracking\")"  -e "gSystem->Load(\"$ONNX_PATH\")" -l -q -b runMatching.C+ | tee matching.log
    RESULTSDIR="Results`cat MatchingConfig.txt`"
    mkdir -p ${RESULTSDIR}
    cp ${MATCHINGRESULTS} "${RESULTSDIR}"

    popd
    echo " Finished matching on `realpath ${OUTDIR}`"

  fi

}

exportMLTrainningData()
{

  if ! [ -f "${OUTDIR}/tempMCHTracks.root" ]; then
    echo " Nothing to export... MCH Tracks not found on `realpath ${OUTDIR}` ..."
    EXITERROR="1"
  fi

  if ! [ -f "${OUTDIR}/mfttracks.root" ]; then
    echo " Nothing to export... MFT Tracks not found on `realpath ${OUTDIR}` ..."
    EXITERROR="1"
  fi

  if ! [ -z ${EXITERROR+x} ]; then exit ; fi

  if [ -d "${OUTDIR}" ]; then
    if ! [ -z ${UPDATECODE+x} ]; then updatecode ; fi

    pushd ${OUTDIR}
    echo "Exporting ML Traning data file on `pwd` ..."
    ## MFT MCH track matching & global muon track fitting:
    root -e 'gSystem->Load("libO2MCHTracking")'  -e "gSystem->Load(\"$ONNX_PATH\")" -l -q -b runMatching.C+ | tee training_data_gen.log
    RESULTSDIR="MLTraining`cat MatchingConfig.txt`"
    mkdir -p ${RESULTSDIR}
    cp training_data_gen.log MLTraining_*.root "${RESULTSDIR}"

    popd
    echo " Finished exporting ML Traning data. File saved on `realpath ${RESULTSDIR}`"
  fi


}


trainML()
{

  if ! [ -f "MLConfigs.xml" ]; then
    echo " Machine Learning configuration file absent..."
    cp MLConfigs.xml "${OUTDIR}"
  fi

  if ! [ -f "${ML_TRAINING_FILE}" ]; then
    echo " ERROR: could not open data file! "
    exit
  fi

  export ML_TYPE=${ML_TYPE:-"Regression"}
  if [ $ML_TEST ]; then
      export ML_NTEST=${ML_NTEST:-"0.1"}
  fi

  if [ -d "${OUTDIR}" ]; then
      if ! [ -z ${UPDATECODE+x} ]; then updatecode ; fi
      pushd ${OUTDIR}

      if ! [ -z ${ML_PYTHON+x} ]; then
	      python3 python_training.py | tee MLtraining.log
      else
	      root -e 'gSystem->Load("libO2MCHTracking")'  -e "gSystem->Load(\"$ONNX_PATH\")" -l -q -b runMatching.C+ | tee MLtraining.log
      fi
  fi

}


runChecks()
{

  if ! [ -f "${OUTDIR}/GlobalMuonTracks.root" ]; then
    echo " Nothing to check... Global Muon Tracks not found on `realpath ${OUTDIR}/GlobalMuonChecks.root` ..."
    EXITERROR="1"
  fi

  if ! [ -f "${OUTDIR}/mfttracks.root" ]; then
    echo " Nothing to check... MFT Tracks not found on `realpath ${OUTDIR}/mfttracks.root` ..."
    EXITERROR="1"
  fi

  if ! [ -z ${EXITERROR+x} ]; then exit ; fi

  if ! [ -z ${UPDATECODE+x} ]; then updatecode ; fi
  pushd ${OUTDIR}
  echo "Checking global muon tracks on `pwd` ..." && \

  ## Check global muon Tracks
  #root -l -q -b GlobalMuonChecks.C+ | tee checks.log
  root -l -q -b CheckThePerformance.C+ | tee performance.log
  RESULTSDIR="Results`cat MatchingConfig.txt`"
  cp ${MATCHINGRESULTS} ${CHECKRESULTS} "${RESULTSDIR}"
  echo " Results moved to `realpath ${RESULTSDIR}`"
  popd
  echo " Finished checking Global muon tracks on `realpath ${OUTDIR}`"

}

SCRIPTDIR=`dirname "$0"`


while [ $# -gt 0 ] ; do
  case $1 in
    -n)
    NEV_="$2";
    shift 2
    ;;
    -j)
    JOBS="$2";
    shift 2
    ;;
    -o)
    OUTDIR="$2";
    shift 2
    ;;
    --npions)
    export NPIONS="$2";
    shift 2
    ;;
    --nmuons)
    export NMUONS="$2";
    shift 2
    ;;
    --njpsis)
    export NJPSI="$2";
    shift 2
    ;;
    --nupsilons)
    export NUPSILON="$2";
    shift 2
    ;;
    -g)
    if [ -z ${GENERATOR+x} ]
    then
      GENERATOR="$2";
    else
      GENERATOR="${GENERATOR}.$2";
    fi
    shift 2
    ;;
    --seed)
    export SEED="$2";
    shift 2
    ;;
    --genMCH)
    GENERATEMCH="1";
    shift 1
    ;;
    --genMFT)
    GENERATEMFT="1";
    shift 1
    ;;
    --shm-segment-size)
    CUSTOM_SHM="--shm-segment-size $2";
    shift 1
    ;;
    --match)
    MATCHING="1";
    shift 1
    ;;
    --matchPlaneZ)
    export MATCHING_PLANEZ="$2";
    shift 2
    ;;
    --InitMFTTracksFromVertexingParameters)
    export INIT_MFT_FROM_VERTEXING="1";
    shift 1
    ;;
    --matchSaveAll)
    export MATCH_SAVE_ALL="1";
    shift 1
    ;;
    --matchFcn)
    export MATCHING_FCN="${2}_";
    shift 2
    ;;
    --cutFcn)
    export MATCHING_CUTFCN="${2}_";
    shift 2
    ;;
    --cutParam0)
    export MATCHING_CUTPARAM0="$2";
    shift 2
    ;;
    --cutParam1)
    export MATCHING_CUTPARAM1="$2";
    shift 2
    ;;
    --cutParam2)
    export MATCHING_CUTPARAM2="$2";
    shift 2
    ;;
    --enableChargeMatchCut)
    export ENABLECHARGEMATCHCUT="1";
    shift 1
    ;;
    --CorrectMatchIgnoreCut)
    export ML_CORRECTMATCHIGNORECUT="1";
    shift 2
    ;;
    --exportTrainingData)
    export ML_EXPORTTRAINDATA="$2";
    shift 2
    ;;
    --onPythonML)
    export ML_PYTHON="1";
    shift 1
    ;;
    --onnxML)
    export ML_ONNX="1";
    shift 1
    ;;
    --weightfile)
    export ML_WEIGHTFILE="`realpath $2`";
    shift 2
    ;;
    --MLScoreCut)
    export ML_SCORECUT="$2";
    shift 2
    ;;
    --train)
    export TRAIN_ML_METHOD="$2";
    shift 2
    ;;
    --mltest)
    export ML_TEST="1";
    shift 1
    ;;
    --ntest)
    export ML_NTEST="$2";
    shift 2
    ;;
    --type)
    export ML_TYPE="$2";
    shift 2
    ;;
    --layout)
    export ML_LAYOUT="$2";
    shift 2
    ;;
    --strategy)
    export ML_TRAINING_STRAT="$2";
    shift 2
    ;;
    --MLoptions)
    export ML_GENERAL_OPT="$2";
    shift 2
    ;;
    --MLModule)
    export ML_MODULE="$2";
    shift 2
    ;;
    --trainingdata)
    export ML_TRAINING_FILE="`realpath $2`";
    shift 2
    ;;
    --bkg)
    export ML_BKG_FILE="`realpath $2`";
    shift 2
    ;;
    --testdata)
    export ML_TESTING_FILE="`realpath $2`";
    shift 2
    ;;
    --testbkg)
    export ML_TESTING_BKG="`realpath $2`";
    shift 2
    ;;
    --convert)
    CONVERT="1";
    shift 1
    ;;
    --check)
    CHECKS="1";
    shift 1
    ;;
    --updatecode)
    UPDATECODE="1";
    shift 1
    ;;
    --verbose)
    export VERBOSEMATCHING="1";
    shift 1
    ;;
    -h|--help)
    Usage
    ;;
    *) echo "Wrong input"; Usage;

  esac
done

# Ensure no enviroment is loaded
#if ! [[ -z "$LOADEDMODULES" ]]
# then
#   echo "Do not run this script with alienv environment loaded. Aborting..."
#   echo "Run '${0##*/} --help'"
#   exit
# fi


if [ -z ${GENERATEMCH+x} ] && [ -z ${GENERATEMFT+x} ] && [ -z ${MATCHING+x} ] && [ -z ${CHECKS+x} ] && [ -z ${ML_EXPORTTRAINDATA+x} ] && [ -z ${TRAIN_ML_METHOD+x} ]
then
  echo "Missing use mode!"
  echo " "
  Usage
fi


if [ -z ${OUTDIR+x} ]; then echo "Missing output dir" ; Usage ; fi
NEV_=${NEV_:-"4"}
JOBS="1" # ${JOBS:-"1"} # Forcing O2 simulation with one worker: necessary to keep event ordering
GENERATOR=${GENERATOR:-"gun0_100GeV"}
CUSTOM_SHM="--shm-segment-size 5000000000"

export MCHGENERATOR=${GENERATOR}
export ALIROOT_OCDB_ROOT=${ALIROOT_OCDB_ROOT:-$HOME/alice/OCDB}
export ONNX_PATH=`locate libonnxruntime.so.1.7.2 | grep local1 | grep alice1`

ALIROOTENV=${ALIROOTENV:-"AliRoot/latest-master-next-root6"}
O2ENV=${O2ENV:-"O2/latest-dev-o2"}
#O2ENV=${O2ENV:-"O2/latest-f754608ed4-o2"}


if ! [ -z ${GENERATEMCH+x} ]; then
    generateMCHTracks ;
fi


if ! [ -z ${GENERATEMFT+x} ]; then
  generateMFTTracks ;
fi

if ! [ -z ${MATCHING+x} ]; then runMatching ; fi
if ! [ -z ${ML_EXPORTTRAINDATA+x} ]; then exportMLTrainningData ; fi
if ! [ -z ${TRAIN_ML_METHOD+x} ]; then trainML ; fi
if ! [ -z ${CHECKS+x} ]; then runChecks ; fi
