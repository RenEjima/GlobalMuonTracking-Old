#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include "MUONMatcher.h"

Float_t EtaToTheta(Float_t arg);

//_________________________________________________________________________________________________
MUONMatcher::MUONMatcher()
{

  const auto grp = o2::parameters::GRPObject::loadFrom("o2sim_grp.root");
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  mGRP.reset(grp);
  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(
    TGeoGlobalMagField::Instance()->GetField());

  double position[3] = {0, 0, -61.4};
  mField_z = field->getBz(position);
  printf("B field z = %f [kGauss]\n", mField_z);

  mMCHTrackExtrap.setField();
  mCutFunc = &MUONMatcher::matchCutDisabled;
  mMatchFunc = &MUONMatcher::matchMFT_MCH_TracksAllParam;
}

//_________________________________________________________________________________________________
void MUONMatcher::LoadAbsorber()
{

  if (!mGeoManager) {
    mGeoManager = new TGeoManager("Matcher Geometry", "Matcher Geometry");
    o2::passive::Cave("CAVE", "Cave").ConstructGeometry();
    o2::passive::Shil("SHIL", "Small angle beam shield").ConstructGeometry();
    o2::passive::Absorber("ABSO", "Absorber").ConstructGeometry();
    mGeoManager->CloseGeometry();
  }
}

//_________________________________________________________________________________________________
void MUONMatcher::Clear()
{

  mMFTTracks.clear();
  mSortedMFTTracks.clear();
  mMFTClusters.clear();
  mMFTTracksROFs.clear();
  mftTrackLabelsIDx.clear();
  mtrackExtClsIDs.clear();
  mftTrackLabels.clear();
  mSortedMCHTracks.clear();
  mSortedMCHTrackLabels.clear();
  mSortedGlobalMuonTracks.clear();
  mSortedGlobalTrackLabels.clear();
  mSortedGlobalMuonTracksExt.clear();
}

//_________________________________________________________________________________________________
void MUONMatcher::loadMCHTracks()
{
  // This function populates mMCHTracks (vector of MCH tracks)
  //

  // For now loading MCH Tracks
  std::string trkFile = "tempMCHTracks.root";
  TFile* trkFileIn = new TFile(trkFile.c_str());
  TTree* mchTrackTree = (TTree*)trkFileIn->Get("treeMCH");
  std::vector<tempMCHTrack> trackMCHVec, *trackMCHVecP = &trackMCHVec;
  mchTrackTree->SetBranchAddress("tempMCHTracks", &trackMCHVecP);
  mNEvents = mchTrackTree->GetEntries();
  mMatchingHelper.nMCHTracks = 0;

  //
  mSortedMCHTracks.clear();
  mSortedMCHTrackLabels.clear();
  mSortedMCHTracks.resize(mNEvents);
  mSortedMCHTrackLabels.resize(mNEvents);

  mSortedMFTTracks.resize(mNEvents);
  mftTrackLabelsIDx.resize(mNEvents);

  for (int event = 0; event < mNEvents; event++) {
    mchTrackTree->GetEntry(event);
    auto MCHTrkID = 0;
    for (auto& mchtrackIn : trackMCHVec) {
      MCHTrack mchTrackOut;
      mchTrackOut.setZ(mchtrackIn.fZ);
      mchTrackOut.setNonBendingCoor(mchtrackIn.fNonBendingCoor);
      mchTrackOut.setNonBendingSlope(mchtrackIn.fThetaX);
      mchTrackOut.setBendingCoor(mchtrackIn.fBendingCoor);
      mchTrackOut.setBendingSlope(mchtrackIn.fThetaY);
      mchTrackOut.setInverseBendingMomentum(mchtrackIn.fInverseBendingMomentum);
      TMatrixD cov(5, 5);
      cov(0, 0) = mchtrackIn.fCovariances[0];
      cov(0, 1) = mchtrackIn.fCovariances[1];
      cov(0, 2) = mchtrackIn.fCovariances[3];
      cov(0, 3) = mchtrackIn.fCovariances[6];
      cov(0, 4) = mchtrackIn.fCovariances[10];

      cov(1, 1) = mchtrackIn.fCovariances[2];
      cov(1, 2) = mchtrackIn.fCovariances[4];
      cov(1, 3) = mchtrackIn.fCovariances[7];
      cov(1, 4) = mchtrackIn.fCovariances[11];

      cov(2, 2) = mchtrackIn.fCovariances[5];
      cov(2, 3) = mchtrackIn.fCovariances[8];
      cov(2, 4) = mchtrackIn.fCovariances[12];

      cov(3, 3) = mchtrackIn.fCovariances[9];
      cov(3, 4) = mchtrackIn.fCovariances[13];

      cov(4, 4) = mchtrackIn.fCovariances[14];

      cov(1, 0) = cov(0, 1);
      cov(2, 0) = cov(0, 2);
      cov(3, 0) = cov(0, 3);
      cov(4, 0) = cov(0, 4);

      cov(2, 1) = cov(1, 2);
      cov(3, 1) = cov(1, 3);
      cov(4, 1) = cov(1, 4);

      cov(3, 2) = cov(2, 3);
      cov(4, 2) = cov(2, 4);

      cov(4, 3) = cov(3, 4);

      mchTrackOut.setCovariances(cov);

      if (mVerbose)
        std::cout << " Loading MCH Track # " << MCHTrkID
                  << "; Label = " << mchtrackIn.fLabel << " in event "
                  << mchtrackIn.fiEv << std::endl;

      if (mchtrackIn.fiEv > mNEvents)
        mNEvents = mchtrackIn.fiEv;

      o2::MCCompLabel thisLabel(mchtrackIn.fLabel, mchtrackIn.fiEv, 0,
                                0); // FIXME: srcID
      mSortedMCHTrackLabels[event].addElement(MCHTrkID, thisLabel);
      mSortedMCHTracks[event].push_back(mchTrackOut);
      MCHTrkID++;
    }
    mMatchingHelper.nMCHTracks += MCHTrkID;
  }

  std::cout << "Loaded " << mMatchingHelper.nMCHTracks << " MCH Tracks in " << mNEvents
            << " events" << std::endl;

  std::ifstream genConfigFile("MatcherGenConfig.txt");
  if (genConfigFile) {
    std::getline(genConfigFile, mMatchingHelper.Generator);
    std::cout << "Generator: " << mMatchingHelper.Generator << std::endl;
  }
}

//_________________________________________________________________________________________________
void MUONMatcher::loadMFTTracksOut()
{
  // Load all MFTTracks and propagate to last MFT Layer;

  std::string trkFile = "mfttracks.root";
  TFile* trkFileIn = new TFile(trkFile.c_str());
  TTree* mftTrackTree = (TTree*)trkFileIn->Get("o2sim");
  std::vector<o2::mft::TrackMFT> trackMFTVec, *trackMFTVecP = &trackMFTVec;
  mftTrackTree->SetBranchAddress("MFTTrack", &trackMFTVecP);

  std::vector<int> trackExtClsVec, *trackExtClsVecP = &trackExtClsVec;
  mftTrackTree->SetBranchAddress("MFTTrackClusIdx", &trackExtClsVecP);

  std::vector<o2::MCCompLabel>* mcLabels = nullptr;
  mftTrackTree->SetBranchAddress("MFTTrackMCTruth", &mcLabels);
  std::vector<o2::itsmft::ROFRecord>* mMFTTracksROFsP = nullptr;
  mftTrackTree->SetBranchAddress("MFTTracksROF", &mMFTTracksROFsP);
  mftTrackTree->GetEntry(0);
  mftTrackLabels = *mcLabels;
  mMFTTracks.swap(trackMFTVec);
  mtrackExtClsIDs.swap(trackExtClsVec);
  mMFTTracksROFs.swap(*mMFTTracksROFsP);

  std::cout << "Loaded " << mMFTTracks.size()
            << " MFT Tracks." << std::endl;
  auto mftTrackID = 0;
  auto nInvalidMFTLabels = 0;
  for (auto& track : mMFTTracks) {
    auto MFTlabel = mftTrackLabels.at(mftTrackID);
    if (MFTlabel.isValid()) {
      auto event = MFTlabel.getEventID();
      if (mLoadMFTOutparameters) {
        track.setParameters(track.getOutParam().getParameters());
        track.setCovariances(track.getOutParam().getCovariances());
        track.setZ(track.getOutParam().getZ());
        track.propagateToZhelix(mMatchingPlaneZ, mField_z);
      } else {
        propagateToNextClusterWithMCS(track, mMatchingPlaneZ);
      }
      mSortedMFTTracks[event].push_back(track);
      mftTrackLabelsIDx[event].push_back(mftTrackID);
    } else {
      nInvalidMFTLabels++;
    }
    mftTrackID++;
  }
  std::cout << " Dropped " << nInvalidMFTLabels << " MFT Tracks with invalid labels (noise)." << std::endl;
  loadMFTClusters();
}

//_________________________________________________________________________________________________
void MUONMatcher::loadMFTClusters()
{

  using o2::itsmft::CompClusterExt;

  constexpr float DefClusErrorRow = 26.88e-4 * 0.5;
  constexpr float DefClusErrorCol = 29.24e-4 * 0.5;
  constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
  constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

  // Geometry and matrix transformations
  std::string inputGeom = "o2sim_geometry.root";
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::mft::GeometryTGeo::Instance();
  gman->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Cluster pattern dictionary
  std::string dictfile = "MFTdictionary.bin";
  o2::itsmft::TopologyDictionary dict;
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    printf("Running with dictionary: %s \n", dictfile.c_str());
    dict.readBinaryFile(dictfile);
  } else {
    printf("Can not run without dictionary !\n");
    return;
  }

  // Clusters

  TFile fileC("mftclusters.root");
  TTree* clsTree = (TTree*)fileC.Get("o2sim");
  std::vector<CompClusterExt> clsVec, *clsVecP = &clsVec;
  clsTree->SetBranchAddress("MFTClusterComp", &clsVecP);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clsLabels = nullptr;
  if (clsTree->GetBranch("MFTClusterMCTruth")) {
    clsTree->SetBranchAddress("MFTClusterMCTruth", &clsLabels);
  } else {
    printf("No Monte-Carlo information in this file\n");
    return;
  }

  int nEntries = clsTree->GetEntries();
  printf("Number of entries in clusters tree %d \n", nEntries);

  clsTree->GetEntry(0);

  int nClusters = (int)clsVec.size();
  printf("Number of clusters %d \n", nClusters);

  auto clusterId = 0;
  for (auto& c : clsVec) {
    auto chipID = c.getChipID();
    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locC;
    float sigmaX2 = DefClusError2Row, sigmaY2 = DefClusError2Col;

    if (pattID != o2::itsmft::CompCluster::InvalidPatternID) {
      sigmaX2 = dict.getErr2X(pattID); // ALPIDE local X coordinate => MFT global X coordinate (ALPIDE rows)
      sigmaY2 = dict.getErr2Z(pattID); // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)
      locC = dict.getClusterCoordinates(c);
    }

    // Transformation to the local --> global
    auto gloC = gman->getMatrixL2G(chipID) * locC;
    // printf("Cluster %5d   chip ID %03d   evn %2d   mctrk %4d   x,y,z  %7.3f
    // %7.3f  %7.3f \n", icls, cluster.getChipID(), evnID, trkID, gloC.X(),
    // gloC.Y(), gloC.Z());

    auto clsPoint2D = o2::math_utils::Point2D<Float_t>(gloC.x(), gloC.y());
    Float_t rCoord = clsPoint2D.R();
    Float_t phiCoord = clsPoint2D.Phi();
    o2::math_utils::bringTo02PiGen(phiCoord);
    int rBinIndex = 0;   // constants::index_table::getRBinIndex(rCoord);
    int phiBinIndex = 0; // constants::index_table::getPhiBinIndex(phiCoord);
    int binIndex =
      0; // constants::index_table::getBinIndex(rBinIndex, phiBinIndex);
    MFTCluster& thisCluster = mMFTClusters.emplace_back(
      gloC.x(), gloC.y(), gloC.z(), phiCoord, rCoord, clusterId, binIndex,
      sigmaX2, sigmaY2, chipID);
    clusterId++;
  }
}

//_________________________________________________________________________________________________
void MUONMatcher::initGlobalTracks()
{
  // Populates Global Muon Track container using MCH track data, unless mMatchSaveAll is set

  if (!mGeoManager) {
    mGeoManager = new TGeoManager("Matcher Geometry", "Matcher Geometry");
    o2::passive::Cave("CAVE", "Cave").ConstructGeometry();
    o2::passive::Shil("SHIL", "Small angle beam shield").ConstructGeometry();
    o2::passive::Absorber("ABSO", "Absorber").ConstructGeometry();
    mGeoManager->CloseGeometry();
  }

  mSortedGlobalMuonTracks.clear();
  mSortedGlobalMuonTracksExt.clear();
  mSortedGlobalMuonTracks.resize(mNEvents);
  mSortedGlobalMuonTracksExt.resize(mNEvents);
  mSortedGlobalTrackLabels.resize(mNEvents);

  if (!mMatchSaveAll) {
    for (auto event = 0; event < mNEvents; event++) {
      for (auto& track : mSortedMCHTracks[event]) {
        mMCHTrackExtrap.extrapToMatchingPlane(&track, mMatchingPlaneZ);
        mSortedGlobalMuonTracks[event].push_back(MCHtoGlobal(track));
      }
      mSortedMCHTracks[event].clear();
    }
  }

  // Populates mMatchingHelper.MatchingCutConfig with mCutParams
  MatchingHelper& helper = mMatchingHelper;
  helper.matchingPlaneZ = mMatchingPlaneZ;

  auto iparam = 0;
  mMatchingHelper.MatchingCutConfig = "";
  for (auto param : mCutParams) {
    mMatchingHelper.MatchingCutConfig = mMatchingHelper.MatchingCutConfig +
                                        "_CutP" + std::to_string(iparam) + "=" +
                                        std::to_string(param);
    iparam++;
  }
  if (mMatchingHelper.MatchingFunction == "") {
    if (mMatchFunc == &MUONMatcher::matchMFT_MCH_TracksXY)
      mMatchingHelper.MatchingFunction = "_matchXY";
    if (mMatchFunc == &MUONMatcher::matchMFT_MCH_TracksXYPhiTanl)
      mMatchingHelper.MatchingFunction = "_matchXYPhiTanl";
    if (mMatchFunc == &MUONMatcher::matchMFT_MCH_TracksAllParam)
      mMatchingHelper.MatchingFunction = "_matchAllParams";
  }

  if (mMatchingHelper.MatchingCutFunc == "") {
    if (mCutFunc == &MUONMatcher::matchCutDisabled)
      helper.MatchingCutFunc = "_cutDisabled";
    if (mCutFunc == &MUONMatcher::matchCutDistance)
      helper.MatchingCutFunc = "_cutDistance";
    if (mCutFunc == &MUONMatcher::matchCutDistanceSigma)
      helper.MatchingCutFunc = "_cutDistanceSigma";
    if (mCutFunc == &MUONMatcher::matchCutDistanceAndAngles)
      helper.MatchingCutFunc = "_cutDistanceAndAngles";
    if (mCutFunc == &MUONMatcher::matchCut3SigmaXYAngles)
      helper.MatchingCutFunc = "_cutDistanceAndAngles3Sigma";
    if (mCutFunc == &MUONMatcher::matchCut3Sigma)
      helper.MatchingCutFunc = "_cut3Sigma";
    if (mCutFunc == &MUONMatcher::matchCutNSigma)
      helper.MatchingCutFunc = "_cutNSigma";
    if (mCutFunc == &MUONMatcher::matchCutVarXYAngles)
      helper.MatchingCutFunc = "_cutDistanceAndAnglesVar";
  }
}

//_________________________________________________________________________________________________
void MUONMatcher::runEventMatching()
{
  // Runs matching over all tracks on a single event
  std::cout << "Number of MCH Tracks = " << mMatchingHelper.nMCHTracks << std::endl;
  std::cout << "Annotation: " << mMatchingHelper.Annotation() << std::endl;
  std::cout << "Running runEventMatching for " << mNEvents << " events"
            << std::endl;

  uint32_t GTrackID = 0;
  auto nMCHTracks = 0;
  auto nGlobalMuonTracksExt = 0;
  for (int event = 0; event < mNEvents; event++) {
    nMCHTracks += (int)mSortedGlobalMuonTracks[event].size();
    if (mVerbose)
      std::cout << "Matching event # " << event << ": " << mSortedGlobalMuonTracks[event].size() << " MCH Tracks" << std::endl;
    GTrackID = 0;

    if (!mMatchSaveAll) {
      for (auto& gTrack : mSortedGlobalMuonTracks[event]) {
        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(GTrackID);
        auto mftTrackID = 0;
        for (auto mftTrack : mSortedMFTTracks[event]) {
          auto MFTlabel = mftTrackLabels.at(mftTrackLabelsIDx[event][mftTrackID]);
          if (matchingCut(gTrack, mftTrack)) {
            gTrack.countCandidate();
            if (MFTlabel.getTrackID() == MCHlabel[0].getTrackID())
              gTrack.setCloseMatch();
            auto chi2 = matchingEval(gTrack, mftTrack);
            if (chi2 < gTrack.getMatchingChi2()) {
              gTrack.setBestMFTTrackMatchID(mftTrackLabelsIDx[event][mftTrackID]);
              gTrack.setMatchingChi2(chi2);
            }
          }
          mftTrackID++;
        }
        GTrackID++;
      }      // /loop over global tracks
    } else { // if matchSaveAll is set
      // store all given matches for performance studies
      // loop MCH tracks
      for (auto& track : mSortedMCHTracks[event]) {
        //printf("BV: MCH track %d \n", GTrackID);
        mMCHTrackExtrap.extrapToVertexWithoutBranson(&track, mMatchingPlaneZ);
        GlobalMuonTrackExt gTrackTmp{MCHtoGlobal(track)};
        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(GTrackID);
        auto mftTrackID = 0;
        for (auto mftTrack : mSortedMFTTracks[event]) {
          //printf("BV: MFT track %d \n", mftTrackID);
          auto MFTlabel = mftTrackLabels.at(mftTrackLabelsIDx[event][mftTrackID]);
          if (MFTlabel.getEventID() == event) {
            if (matchingCut(gTrackTmp, mftTrack)) {
              GlobalMuonTrackExt gTrack{MCHtoGlobal(track)};
              gTrack.setParametersMCH(gTrack.getParameters());
              gTrack.setCovariancesMCH(gTrack.getCovariances());
              gTrack.countCandidate();
              //printf("BV: match MCH %d MFT %d \n", GTrackID, mftTrackID);
              if (MFTlabel.getTrackID() == MCHlabel[0].getTrackID()) {
                gTrack.setCloseMatch();
              }
              auto chi2 = matchingEval(gTrack, mftTrack);
              gTrack.setBestMFTTrackMatchID(mftTrackLabelsIDx[event][mftTrackID]);
              gTrack.setMatchingChi2(chi2);
              gTrack.setParametersMFT(mftTrack.getParameters());
              gTrack.setCovariancesMFT(mftTrack.getCovariances());
              gTrack.setMCHTrackID(GTrackID);
              mSortedGlobalMuonTracksExt[event].push_back(gTrack);
            } // matching cut
          }   // end event match, MFT
          mftTrackID++;
        } // end loop MFT tracks
        GTrackID++;
      } // end loop MCH tracks
    }   // end match save all
    nGlobalMuonTracksExt += (int)mSortedGlobalMuonTracksExt[event].size();
  } // /loop over events

  if (!mMatchSaveAll) {
    std::cout << "Finished runEventMatching on " << nMCHTracks << " MCH Tracks on " << mNEvents << " events"
              << std::endl;
  } else {
    std::cout << "Finished runEventMatching with matchSaveAll option " << nGlobalMuonTracksExt << " global muon tracks."
              << std::endl;
  }
  finalize();
  if (mTMVAReader)
    EvaluateML();
}

//_________________________________________________________________________________________________
void MUONMatcher::MLClassification(std::string input_name, std::string trainingfile,
                                   std::string trainingstr)
{

  TMVA::Tools::Instance();

  std::cout << "==> Start TMVAClassification" << std::endl;
  TTree* signalTree;
  TTree* background;

  TFile* input(0);
  if (!gSystem->AccessPathName(trainingfile.c_str())) {
    input = TFile::Open(trainingfile.c_str());
  }
  if (!input) {
    std::cout << "ERROR: could not open data file" << std::endl;
    exit(1);
  }
  std::cout << "--- TMVAClassification           : Using training input file: " << input->GetName() << std::endl;

  if (gSystem->Getenv("ML_BKG_FILE")) {
    std::string bkgfile = gSystem->Getenv("ML_BKG_FILE");
    TFile* input_bkg(0);
    if (!gSystem->AccessPathName(trainingfile.c_str())) {
      input_bkg = TFile::Open(bkgfile.c_str());
    }
    if (!input_bkg) {
      std::cout << "ERROR: could not open background data file" << std::endl;
      exit(1);
    }
    std::cout << "--- TMVAClassification           : Using background file: " << input_bkg->GetName() << std::endl;

    // Register the training and test trees
    signalTree = (TTree*)input->Get("matchTree");
    background = (TTree*)input_bkg->Get("matchTree");

  } else {
    signalTree = (TTree*)input->Get("Signal_tree");
    background = (TTree*)input->Get("Bkg_tree");
  }

  // Let's initialize the factory object (analysis class)...:
  // The first argument is the base of the name of all the
  // weightfiles in the directory weight/
  //
  // The second argument is the output file for the training results.
  // The weightfile name will be the the combined strings of the two arguments
  //
  // In the third argument we can set the analysis type: classification or regression

  TString methodname = input_name.c_str();

  std::size_t rpos = trainingfile.rfind('/');
  std::size_t pos = trainingfile.find(".root");
  methodname += "_" + trainingfile.substr(rpos + 1, (pos - (rpos + 1)));
  TString outfileName(methodname + ".root");
  TFile* outputFile = TFile::Open(outfileName, "RECREATE");

  std::string MLMethodType = gSystem->Getenv("TRAIN_ML_METHOD");

  TMVA::Factory* factory = new TMVA::Factory(
    "Classification_" + MLMethodType, outputFile,
    "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification"); //TODO check these transformations options

  if (MLMethodType == "BDTG" or MLMethodType == "BDT" or MLMethodType == "BDTB" or MLMethodType == "BDTD" or MLMethodType == "BDTF") {
    MLMethodType = "BDT";
  } else if (MLMethodType == "MLP" or MLMethodType == "MLPBFGS" or MLMethodType == "MLPBNN") {
    MLMethodType = "MLP";
  } else if (MLMethodType == "Cuts" or MLMethodType == "CutsD" or MLMethodType == "CutsSA" or
             MLMethodType == "CutsGA" or MLMethodType == "CutsPCA") {
    MLMethodType = "Cuts";
  } else if (MLMethodType == "FDA_GA" or MLMethodType == "FDA_GAMT" or MLMethodType == "FDA_SA" or
             MLMethodType == "FDA_MC" or MLMethodType == "FDA_MT" or MLMethodType == "FDA_MCMT") {
    MLMethodType = "FDA";
  } else if (MLMethodType == "Likelihood" or MLMethodType == "LikelihoodD" or MLMethodType == "LikelihoodPCA" or
             MLMethodType == "LikelihoodKDE" or MLMethodType == "LikelihoodMIX") {
    MLMethodType = "Likelihood";
  } else if (MLMethodType == "PDERS" or MLMethodType == "PDERSD" or MLMethodType == "PDERSPCA") {
    MLMethodType = "PDERS";
  } else if (MLMethodType == "PDEFoam" or MLMethodType == "PDEFoamBoost") {
    MLMethodType = "PDEFoam";
  } else if (MLMethodType == "Fisher" or MLMethodType == "FisherG" or MLMethodType == "BoostedFisher") {
    MLMethodType = "Fisher";
  }

  // Dataloader object - this will handle the data (The argument also defines the name of the directory containing the weights' file)
  TMVA::DataLoader* dataloader = new TMVA::DataLoader("trainedML");

  for (std::size_t i = 0; i < mNInputFeatures; i++) {
    dataloader->AddVariable(mMLInputFeaturesName[i], mMLInputFeaturesName[i], "units", 'F');
  }

  // global event weights per tree (see below for setting event-wise weights)
  Double_t signalWeight = 1.0;
  Double_t backgroundWeight = 1.0;

  // Apply additional cuts on the signal and background samples (can be different)
  TCut mycutSig = ""; // for example: TCut mycutSig = "abs(var1)<0.5 && abs(var2-0.5)<1";
  TCut mycutBkg = ""; // for example: TCut mycutBkg = "abs(var1)<0.5";

  if (gSystem->Getenv("ML_TEST")) {

    if (gSystem->Getenv("ML_TESTING_FILE")) {
      std::string testingfile = gSystem->Getenv("ML_TESTING_FILE");
      std::string test_bkg = gSystem->Getenv("ML_TESTING_BKG");
      TFile* test_input(0);
      TFile* test_input2(0);
      if (!gSystem->AccessPathName(testingfile.c_str())) {
        test_input = TFile::Open(testingfile.c_str());
      }
      if (!test_input) {
        std::cout << "ERROR: could not open testing data file" << std::endl;
        exit(1);
      }
      std::cout << "--- TMVARegression           : Using testing input file: "
                << test_input->GetName() << std::endl;

      if (!gSystem->AccessPathName(test_bkg.c_str())) {
        test_input2 = TFile::Open(test_bkg.c_str());
      }
      if (!test_input2) {
        std::cout << "ERROR: could not open background testing data file" << std::endl;
        exit(1);
      }
      std::cout << "--- TMVARegression           : Using background testing file: "
                << test_input2->GetName() << std::endl;
      // Register the regression test tree (for now, must use same format as
      // training tree)
      TTree* SigTreeTest = (TTree*)test_input->Get("matchTree");
      TTree* BkgTreeTest = (TTree*)test_input2->Get("matchTree");

      dataloader->AddSignalTree(signalTree, signalWeight, Types::kTraining);
      dataloader->AddBackgroundTree(background, backgroundWeight, Types::kTraining);
      dataloader->AddSignalTree(SigTreeTest, signalWeight, Types::kTesting);
      dataloader->AddBackgroundTree(BkgTreeTest, backgroundWeight, Types::kTesting);

      dataloader->PrepareTrainingAndTestTree(
        mycutSig, mycutBkg, "SplitMode=Random:NormMode=NumEvents:!V");

    } else {
      float ntest = atof((gSystem->Getenv("ML_NTEST")));
      int nentries_sig = signalTree->GetEntries();
      int nentries_bkg = background->GetEntries();
      int train_signal;
      int train_bkg;

      if (ntest < 1) {
        train_signal = (1 - ntest) * nentries_sig;
        train_bkg = (1 - ntest) * nentries_bkg;
      } else {
        train_signal = nentries_sig - ntest;
        train_bkg = nentries_bkg - ntest;
      }

      dataloader->AddSignalTree(signalTree, signalWeight);
      dataloader->AddBackgroundTree(background, backgroundWeight);

      // Set individual event weights (the variables must exist in the original TTree)
      // -  for signal    : `dataloader->SetSignalWeightExpression    ("weight1*weight2");`
      // -  for background: `dataloader->SetBackgroundWeightExpression("weight1*weight2");`
      // dataloader->SetBackgroundWeightExpression( "weight" );

      dataloader->PrepareTrainingAndTestTree(
        mycutSig, mycutBkg,
        "nTrain_Signal=" + std::to_string(train_signal) +
          ":nTrain_Background=" + std::to_string(train_bkg) +
          ":SplitMode=Random:NormMode="
          "NumEvents:!V"); // nTest_Regression=0
                           // uses all events left for testing
    }
  } else { // i.e. just training the ML, wo testing and evaluation
           // You can now add the tree to the dataloader:

    dataloader->AddSignalTree(signalTree, signalWeight, Types::kTraining);
    dataloader->AddBackgroundTree(background, backgroundWeight, Types::kTraining);
    dataloader->PrepareTrainingAndTestTree(
      mycutSig, mycutBkg, "SplitMode=Random:NormMode=NumEvents:!V");
  }

  // Book (SAVE) the method
  factory->BookMethod(dataloader,
                      TMVA::Types::Instance().GetMethodType(MLMethodType),
                      methodname, trainingstr);
  // --------------------------------------------------------------------------------------------------
  std::ofstream info("time_" + methodname + ".txt");
  TStopwatch* timewatch = new TStopwatch();
  // Train MVAs using the set of training events
  factory->TrainAllMethods();

  if (gSystem->Getenv("ML_TEST")) {
    // Evaluate all MVAs using the set of test events
    factory->TestAllMethods();

    // Evaluate and compare performance of all configured MVAs
    factory->EvaluateAllMethods();
  }
  info << "\n END OF THE TRAINING \n";
  info << " ~CPU time (s): " << timewatch->CpuTime() << "\n\n"
       << " ~Real time(s): " << timewatch->RealTime() << "\n\n";
  // --------------------------------------------------------------

  // Save the output
  outputFile->Close();

  std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
  std::cout << "==> TMVAClassification is done!" << std::endl;

  delete factory;
  delete dataloader;

  // Launch the GUI for the root macros
  if (!gROOT->IsBatch())
    TMVA::TMVARegGui(outfileName);
}

//_________________________________________________________________________________________________
void MUONMatcher::MLRegression(std::string input_name, std::string trainingfile,
                               std::string trainingstr)
{

  TMVA::Tools::Instance();

  std::cout << "==> Start TMVARegression" << std::endl;

  TFile* input(0);
  if (!gSystem->AccessPathName(trainingfile.c_str())) {
    input = TFile::Open(trainingfile.c_str());
  }
  if (!input) {
    std::cout << "ERROR: could not open data file" << std::endl;
    exit(1);
  }
  std::cout << "--- TMVARegression           : Using training input file: " << input->GetName() << std::endl;

  // Register the regression tree
  TTree* regTree = (TTree*)input->Get("matchTree");

  // Let's initialize the factory object (analysis class)...:
  // The first argument is the base of the name of all the
  // weightfiles in the directory weight/
  //
  // The second argument is the output file for the training results.
  // The weightfile name will be the the combined strings of the two arguments
  //
  // In the third argument we can set the analysis type: classification or regression

  TString methodname = input_name.c_str();

  std::size_t rpos = trainingfile.rfind('/');
  std::size_t pos = trainingfile.find(".root");
  methodname += "_" + trainingfile.substr(rpos + 1, (pos - (rpos + 1)));
  TString outfileName(methodname + ".root");
  TFile* outputFile = TFile::Open(outfileName, "RECREATE");

  std::string MLMethodType = gSystem->Getenv("TRAIN_ML_METHOD");
  TMVA::Factory* factory = new TMVA::Factory(
    "Regression_" + MLMethodType, outputFile,
    "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression");

  if (MLMethodType == "DNN") {
    MLMethodType = "DL";
  } else if (MLMethodType == "BDTG" or MLMethodType == "BDT") {
    MLMethodType = "BDT";
  } else if (MLMethodType == "FDA_GA" or MLMethodType == "FDA_GAMT" or
             MLMethodType == "FDA_MC" or MLMethodType == "FDA_MT") {
    MLMethodType = "FDA";
  }

  // Dataloader object - this will handle the data (The argument also defines the name of the directory containing the weights' file)
  TMVA::DataLoader* dataloader = new TMVA::DataLoader("trainedML");

  for (std::size_t i = 0; i < mNInputFeatures; i++) {
    dataloader->AddVariable(mMLInputFeaturesName[i], mMLInputFeaturesName[i], "units", 'F');
  }

  // Add the variable carrying the regression target
  dataloader->AddTarget("Truth");

  // global event weights per tree (see below for setting event-wise weights)
  Double_t regWeight = 1.0;

  // Options for training (and testing):
  // Apply additional cuts on the signal and background samples (can be
  // different)
  TCut mycut = ""; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";

  if (gSystem->Getenv("ML_TEST")) {

    if (gSystem->Getenv("ML_TESTING_FILE")) {
      std::string testingfile = gSystem->Getenv("ML_TESTING_FILE");
      TFile* test_input(0);
      if (!gSystem->AccessPathName(testingfile.c_str())) {
        test_input = TFile::Open(testingfile.c_str());
      }
      if (!test_input) {
        std::cout << "ERROR: could not open testing data file" << std::endl;
        exit(1);
      }
      std::cout << "--- TMVARegression           : Using testing input file: "
                << test_input->GetName() << std::endl;
      // Register the regression test tree (for now, must use same format as
      // training tree)
      TTree* testTree = (TTree*)test_input->Get("matchTree");
      dataloader->AddRegressionTree(regTree, regWeight, Types::kTraining);
      dataloader->AddRegressionTree(testTree, regWeight, Types::kTesting);

      dataloader->PrepareTrainingAndTestTree(
        mycut, "SplitMode=Random:NormMode=NumEvents:!V");
    } else {
      float ntest = atof((gSystem->Getenv("ML_NTEST")));
      int nentries = regTree->GetEntries();
      int train_samples;
      if (ntest < 1) {
        train_samples = (1 - ntest) * nentries;
      } else {
        train_samples = nentries - ntest;
      }
      dataloader->AddRegressionTree(regTree, regWeight);
      dataloader->PrepareTrainingAndTestTree(
        mycut,
        "nTrain_Regression=" + std::to_string(train_samples) +
          ":nTest_Regression=0:"
          "SplitMode=Random:NormMode="
          "NumEvents:!V"); // nTest_Regression=0
                           // uses all events left for testing
    }
  } else { // i.e. just training the ML, wo testing and evaluation
    // You can now add the tree to the dataloader:
    dataloader->AddRegressionTree(regTree, regWeight, Types::kTraining);
    dataloader->PrepareTrainingAndTestTree(
      mycut, "SplitMode=Random:NormMode=NumEvents:!V");
  }
  // Book (SAVE) the method
  factory->BookMethod(dataloader,
                      TMVA::Types::Instance().GetMethodType(MLMethodType),
                      methodname, trainingstr);
  // --------------------------------------------------------------------------------------------------
  std::ofstream info("time_" + methodname + ".txt");
  TStopwatch* timewatch = new TStopwatch();
  // Train MVAs using the set of training events
  factory->TrainAllMethods();

  if (gSystem->Getenv("ML_TEST")) {
    // Evaluate all MVAs using the set of test events
    factory->TestAllMethods();

    // Evaluate and compare performance of all configured MVAs
    factory->EvaluateAllMethods();
  }
  info << "\n END OF THE TRAINING \n";
  info << " ~CPU time (s): " << timewatch->CpuTime() << "\n\n"
       << " ~Real time(s): " << timewatch->RealTime() << "\n\n";
  // --------------------------------------------------------------

  // Save the output
  outputFile->Close();

  std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
  std::cout << "==> TMVARegression is done!" << std::endl;

  delete factory;
  delete dataloader;

  // Launch the GUI for the root macros
  if (!gROOT->IsBatch())
    TMVA::TMVARegGui(outfileName);
}

//_________________________________________________________________________________________________
bool MUONMatcher::printMatchingPlaneView(int event, int MCHTrackID)
{

  if (MCHTrackID > (int)mSortedGlobalMuonTracks[event].size() or event >= (int)mSortedGlobalMuonTracks.size()) {
    // Out of range
    return false;
  }

  auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(MCHTrackID);
  auto MCHTrack = mSortedGlobalMuonTracks[event][MCHTrackID];

  auto localBestMFTTrack = -1;
  auto localBestMatchChi2 = 1e100;
  auto localCorrectMFTMatch = -1;

  std::vector<double> xPositions;
  std::vector<double> yPositions;
  std::vector<std::string> pointsColors;

  xPositions.emplace_back(MCHTrack.getX());
  yPositions.emplace_back(MCHTrack.getY());
  pointsColors.emplace_back("orange");
  std::string matchParamStr, correctParamStr, mchParamStr, matchCovStr,
    correctCovStr, mchCovStr;
  mchParamStr = getParamString(MCHTrack);
  mchCovStr = getCovString(MCHTrack);

  auto mftTrackID = 0, nMFTTracks = 0;
  for (auto mftTrack : mSortedMFTTracks[event]) {
    auto MFTlabel = mftTrackLabels.at(mftTrackLabelsIDx[event][mftTrackID]);
    if (MFTlabel.getEventID() == event) {
      nMFTTracks++;
      xPositions.emplace_back(mftTrack.getX());
      yPositions.emplace_back(mftTrack.getY());
      pointsColors.emplace_back("black");
      if (matchingCut(MCHTrack, mftTrack)) {
        pointsColors.back() = "blue";
        MCHTrack.countCandidate();
        if (MFTlabel.getTrackID() == MCHlabel[0].getTrackID()) {
          MCHTrack.setCloseMatch();
          pointsColors.back() = "magenta"; // Close match
          localCorrectMFTMatch = pointsColors.size();
          correctParamStr = getParamString(mftTrack);
          correctCovStr = getCovString(mftTrack);
        }
        auto chi2 = matchingEval(MCHTrack, mftTrack);
        if (chi2 < MCHTrack.getMatchingChi2()) {
          MCHTrack.setBestMFTTrackMatchID(mftTrackLabelsIDx[event][mftTrackID]);
          MCHTrack.setMatchingChi2(chi2);
          localBestMFTTrack = pointsColors.size();
          localBestMatchChi2 = chi2;
          matchParamStr = getParamString(mftTrack);
          matchCovStr = getCovString(mftTrack);
        }
      } else {
        if (MFTlabel.getTrackID() == MCHlabel[0].getTrackID()) {
          pointsColors.back() = "violet"; // far match
          correctParamStr = getParamString(mftTrack);
          correctCovStr = getCovString(mftTrack);
        }
      }
    }
    mftTrackID++;
  }

  if (localBestMFTTrack > -1) {
    if (mTMVAReader) {
      if (localBestMatchChi2 < -mMLScoreCut) {
        pointsColors[localBestMFTTrack - 1] = (localCorrectMFTMatch == localBestMFTTrack) ? "green" : "red";
      }
    } else {
      pointsColors[localBestMFTTrack - 1] = (localCorrectMFTMatch == localBestMFTTrack) ? "green" : "red";
    }
  }

  std::vector<double> xPositionsBlack, yPositionsBlack, xPositionsBlue,
    yPositionsBlue, xPositionsGreen, yPositionsGreen, xPositionsRed,
    yPositionsRed, xPositionsOrange, yPositionsOrange, xPositionsMagenta,
    yPositionsMagenta, xPositionsViolet, yPositionsViolet;

  if (mVerbose) {
    std::cout << " printMatchingPlaneView: " << std::endl;
  }

  for (std::size_t i = 0; i < pointsColors.size(); i++) {
    auto x = xPositions[i];
    auto y = yPositions[i];
    auto color = pointsColors[i];

    if (mVerbose) {
      std::cout << " x = " << x << " y = " << y;
      std::cout << " color = " << color << std::endl;
    }

    if (color == "blue") { // Candidate
      xPositionsBlue.emplace_back(x);
      yPositionsBlue.emplace_back(y);
    }

    if (color == "red") { // Fake MFT Track
      xPositionsRed.emplace_back(x);
      yPositionsRed.emplace_back(y);
    }

    if (color == "black") { // Excluded by cut function
      xPositionsBlack.emplace_back(x);
      yPositionsBlack.emplace_back(y);
    }

    if (color == "orange") { // MCH track
      xPositionsOrange.emplace_back(x);
      yPositionsOrange.emplace_back(y);
    }

    if (color == "green") { // Correct MFT match
      xPositionsGreen.emplace_back(x);
      yPositionsGreen.emplace_back(y);
    }

    if (color == "magenta") { // Missed
      xPositionsMagenta.emplace_back(x);
      yPositionsMagenta.emplace_back(y);
    }

    if (color == "violet") {
      xPositionsViolet.emplace_back(x);
      yPositionsViolet.emplace_back(y);
    }
  }

  TCanvas* canvasMatchingPlane =
    new TCanvas((std::string("cMatchingPlane_eV" + std::to_string(event) + std::string("_trk") + std::to_string(MCHTrackID))).c_str(),
                "Matching Plane View", 1000, 1000);
  canvasMatchingPlane->SetFillStyle(4000);
  canvasMatchingPlane->SetFrameLineWidth(3);
  TMultiGraph* MultiGraph_MatchingPlane = new TMultiGraph();
  MultiGraph_MatchingPlane->SetName("MatchingPlaneView");

  auto legend = new TLegend(0.1, 0.73, 0.3, 0.9);
  legend->SetFillColorAlpha(kWhite, 0.);
  legend->SetBorderSize(2);

  TPaveText* pt;

  if (xPositionsBlack.size()) {
    TGraph* TGBlack = new TGraph(xPositionsBlack.size(), &xPositionsBlack[0],
                                 &yPositionsBlack[0]);
    TGBlack->SetTitle("No candidate");
    TGBlack->SetName("Not_tested");
    TGBlack->GetXaxis()->SetTitle("x");
    TGBlack->SetMarkerStyle(kFullCircle);
    TGBlack->SetMarkerColor(kBlack);
    TGBlack->SetMarkerSize(2);
    TGBlack->SetLineWidth(0);
    MultiGraph_MatchingPlane->Add(TGBlack);
    TGBlack->Draw();
  }

  if (xPositionsBlue.size()) {
    TGraph* TGBlue = new TGraph(xPositionsBlue.size(), &xPositionsBlue[0],
                                &yPositionsBlue[0]);
    TGBlue->SetTitle("Candidate");
    TGBlue->SetName("Candidate");
    TGBlue->GetXaxis()->SetTitle("x");
    TGBlue->SetMarkerStyle(kFullCircle);
    TGBlue->SetMarkerColor(kBlue);
    TGBlue->SetMarkerSize(2);
    TGBlue->SetLineWidth(0);
    MultiGraph_MatchingPlane->Add(TGBlue);
    legend->AddEntry(TGBlue);
  }

  if (xPositionsRed.size()) {
    TGraph* TGRed =
      new TGraph(xPositionsRed.size(), &xPositionsRed[0], &yPositionsRed[0]);
    TGRed->SetTitle("Fake Match");
    TGRed->SetName("FakeMatch");
    TGRed->GetXaxis()->SetTitle("x");
    TGRed->SetMarkerStyle(kFullCircle);
    TGRed->SetMarkerColor(kRed);
    TGRed->SetMarkerSize(3);
    TGRed->SetLineWidth(0);
    MultiGraph_MatchingPlane->Add(TGRed);
    legend->AddEntry(TGRed);
  }

  if (xPositionsGreen.size()) {
    TGraph* TGGreen = new TGraph(xPositionsGreen.size(), &xPositionsGreen[0],
                                 &yPositionsGreen[0]);
    TGGreen->SetTitle("CorrectMatch");
    TGGreen->SetName("CorrectMatch");
    TGGreen->GetXaxis()->SetTitle("x");
    TGGreen->SetMarkerStyle(kFullCircle);
    TGGreen->SetMarkerColor(kGreen);
    TGGreen->SetMarkerSize(3);
    TGGreen->SetLineWidth(0);
    MultiGraph_MatchingPlane->Add(TGGreen);
    legend->AddEntry(TGGreen);
  }

  if (xPositionsMagenta.size()) {
    TGraph* TGMagenta = new TGraph(
      xPositionsMagenta.size(), &xPositionsMagenta[0], &yPositionsMagenta[0]);
    TGMagenta->SetTitle("Close Match");
    TGMagenta->SetName("CloseMatch");
    TGMagenta->GetXaxis()->SetTitle("x");
    TGMagenta->SetMarkerStyle(kFullCircle);
    TGMagenta->SetMarkerColor(kMagenta);
    TGMagenta->SetMarkerSize(3);
    TGMagenta->SetLineWidth(0);
    MultiGraph_MatchingPlane->Add(TGMagenta);
    legend->AddEntry(TGMagenta);
  }

  if (xPositionsViolet.size()) {
    TGraph* TGViolet = new TGraph(xPositionsViolet.size(), &xPositionsViolet[0],
                                  &yPositionsViolet[0]);
    TGViolet->SetTitle("Far Match (missed)");
    TGViolet->SetName("FarMatch");
    TGViolet->GetXaxis()->SetTitle("x");
    TGViolet->SetMarkerStyle(kFullSquare);
    TGViolet->SetMarkerColor(kViolet);
    TGViolet->SetMarkerSize(2);
    TGViolet->SetLineWidth(0);
    MultiGraph_MatchingPlane->Add(TGViolet);
    legend->AddEntry(TGViolet);
  }

  TGraph* TGOrange = new TGraph(xPositionsOrange.size(), &xPositionsOrange[0],
                                &yPositionsOrange[0]);
  TGOrange->SetTitle("MCHTrack");
  TGOrange->SetName("MCHTrack");
  TGOrange->GetXaxis()->SetTitle("x");
  TGOrange->SetMarkerStyle(kFullCircle);
  TGOrange->SetMarkerColor(kOrange);
  TGOrange->SetMarkerSize(3);
  TGOrange->SetLineWidth(0);
  TGOrange->SetFillColorAlpha(kWhite, 0.);
  MultiGraph_MatchingPlane->Add(TGOrange);
  legend->AddEntry(TGOrange);

  MultiGraph_MatchingPlane->GetXaxis()->SetTitle("x [cm]");
  MultiGraph_MatchingPlane->GetYaxis()->SetTitle("y [cm]");
  MultiGraph_MatchingPlane->GetYaxis()->SetTitleOffset(1.25);
  MultiGraph_MatchingPlane->SetTitle("MatchingPlaneView");

  auto rOuterMatchingPlane = TMath::Abs(mMatchingPlaneZ * 14.35 / -77.5);
  auto rInnerMatchingPlane = TMath::Abs(mMatchingPlaneZ * 3.9 / -77.5);
  auto rMargin = 1.4;
  MultiGraph_MatchingPlane->GetXaxis()->SetLimits(
    -rMargin * rOuterMatchingPlane, rMargin * rOuterMatchingPlane);
  MultiGraph_MatchingPlane->SetMinimum(-rMargin * rOuterMatchingPlane);
  MultiGraph_MatchingPlane->SetMaximum(rMargin * rOuterMatchingPlane);
  MultiGraph_MatchingPlane->GetYaxis()->SetLimits(
    -rMargin * rOuterMatchingPlane, rMargin * rOuterMatchingPlane);
  gPad->Modified();
  MultiGraph_MatchingPlane->Draw("LP same");

  legend->Draw();
  canvasMatchingPlane->Update();
  TEllipse* outerR =
    new TEllipse(0, 0, rOuterMatchingPlane, rOuterMatchingPlane);
  outerR->SetLineWidth(2);
  outerR->SetFillColorAlpha(kWhite, 0.);
  outerR->Draw();

  TEllipse* innerR =
    new TEllipse(0, 0, rInnerMatchingPlane, rInnerMatchingPlane);
  innerR->SetLineWidth(2);
  innerR->SetFillColorAlpha(kWhite, 0.);
  innerR->Draw();

  pt = new TPaveText(0.1, 0.918, 0.9, 0.995, "NDC");
  pt->SetBorderSize(0);
  pt->SetFillStyle(4000);
  std::string topLabel = "Matching Plane View: Event " + std::to_string(event) + ", MCH Track" + std::to_string(MCHTrackID);
  pt->AddText(topLabel.c_str());
  pt->AddText(mMatchingHelper.Annotation().c_str());
  pt->Draw();

  pt = new TPaveText(0.3, 0.8, 0.9, 0.9, "NDC");
  pt->SetBorderSize(2);
  pt->SetFillStyle(4000);
  pt->AddText(("MCH: " + mchParamStr).c_str());
  pt->AddText((mchCovStr).c_str());

  if (xPositionsRed.size()) {
    pt->AddText(("Fake: " + matchParamStr).c_str());
    pt->AddText((matchCovStr).c_str());
  }

  if (xPositionsGreen.size()) {
    pt->AddText(("Correct: " + matchParamStr).c_str());
    pt->AddText((matchCovStr).c_str());
  }

  if (xPositionsMagenta.size()) {
    pt->AddText(("Close match: " + correctParamStr).c_str());
    pt->AddText((correctCovStr).c_str());
  }

  if (xPositionsViolet.size()) {
    pt->AddText(("Far match: " + correctParamStr).c_str());
    pt->AddText((correctCovStr).c_str());
  }
  pt->Draw();

  pt = new TPaveText(0.11, 0.12, 0.4, 0.15, "NDC");
  pt->SetBorderSize(0);
  pt->SetFillStyle(4000);
  pt->AddText(("nMFTTracks = " + std::to_string(nMFTTracks)).c_str());
  pt->Draw();

  canvasMatchingPlane->Draw();
  canvasMatchingPlane->SaveAs((std::string("MatchingPlane_eV" + std::to_string(event) + "_MCHtrk" + std::to_string(MCHTrackID) + ".png")).c_str());

  return true;
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchingCut(const MCHTrackConv& mchTrack,
                              const MFTTrack& mftTrack)
{

  if (mCustomCutFunc) {
    return (*mCustomCutFunc)(mchTrack, mftTrack);
  }

  return (this->*mCutFunc)(mchTrack, mftTrack);
}

//_________________________________________________________________________________________________
double MUONMatcher::matchingEval(const MCHTrackConv& mchTrack,
                                 const MFTTrack& mftTrack)
{

  if (mCustomMatchFunc) {
    return (*mCustomMatchFunc)(mchTrack, mftTrack);
  }

  return (this->*mMatchFunc)(mchTrack, mftTrack);
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCutDistance(const MCHTrackConv& mchTrack,
                                   const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  return distance < mCutParams[0];
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCutDistanceAndAngles(const MCHTrackConv& mchTrack,
                                            const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto dPhi = TMath::Abs(mchTrack.getPhi() - mftTrack.getPhi());
  auto dTheta =
    TMath::Abs(EtaToTheta(mchTrack.getEta()) - EtaToTheta(mftTrack.getEta()));
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  return (distance < mCutParams[0]) and (dPhi < mCutParams[1]) and
         (dTheta < mCutParams[2]);
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCutDistanceSigma(const MCHTrackConv& mchTrack,
                                        const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  auto cutDistance = mCutParams[0] *
                     TMath::Sqrt(mchTrack.getSigma2X() + mchTrack.getSigma2Y());
  return distance < cutDistance;
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCut3SigmaXYAngles(const MCHTrackConv& mchTrack,
                                         const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto dPhi = mchTrack.getPhi() - mftTrack.getPhi();
  auto dTanl = TMath::Abs(mchTrack.getTanl() - mftTrack.getTanl());
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  auto cutDistance = 3 * TMath::Sqrt(mchTrack.getSigma2X() + mchTrack.getSigma2Y());
  auto cutPhi = 3 * TMath::Sqrt(mchTrack.getSigma2Phi() + mftTrack.getSigma2Phi());
  auto cutTanl = 3 * TMath::Sqrt(mchTrack.getSigma2Tanl() + mftTrack.getSigma2Tanl());
  return (distance < cutDistance) and (dPhi < cutPhi) and (dTanl < cutTanl);
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCut3Sigma(const MCHTrackConv& mchTrack,
                                 const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto dPhi = mchTrack.getPhi() - mftTrack.getPhi();
  auto dTanl = TMath::Abs(mchTrack.getTanl() - mftTrack.getTanl());
  auto dInvQPt = TMath::Abs(mchTrack.getInvQPt() - mftTrack.getInvQPt());
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  auto cutDistance = 3 * TMath::Sqrt(mchTrack.getSigma2X() + mchTrack.getSigma2Y());
  auto cutPhi = 3 * TMath::Sqrt(mchTrack.getSigma2Phi() + mftTrack.getSigma2Phi());
  auto cutTanl = 3 * TMath::Sqrt(mchTrack.getSigma2Tanl() + mftTrack.getSigma2Tanl());
  auto cutInvQPt = 3 * TMath::Sqrt(mchTrack.getSigma2InvQPt() + mftTrack.getSigma2InvQPt());
  return (distance < cutDistance) and (dPhi < cutPhi) and (dTanl < cutTanl) and (dInvQPt < cutInvQPt);
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCutNSigma(const MCHTrackConv& mchTrack,
                                 const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto dPhi = mchTrack.getPhi() - mftTrack.getPhi();
  auto dTanl = TMath::Abs(mchTrack.getTanl() - mftTrack.getTanl());
  auto dInvQPt = TMath::Abs(mchTrack.getInvQPt() - mftTrack.getInvQPt());
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  auto cutDistance = mCutParams[0] * TMath::Sqrt(mchTrack.getSigma2X() + mchTrack.getSigma2Y());
  auto cutPhi = mCutParams[0] * TMath::Sqrt(mchTrack.getSigma2Phi()) + mftTrack.getSigma2Phi();
  auto cutTanl = mCutParams[0] * TMath::Sqrt(mchTrack.getSigma2Tanl() + mftTrack.getSigma2Tanl());
  auto cutInvQPt = mCutParams[0] * TMath::Sqrt(mchTrack.getSigma2InvQPt() + mftTrack.getSigma2InvQPt());
  return (distance < cutDistance) and (dPhi < cutPhi) and (dTanl < cutTanl) and (dInvQPt < cutInvQPt);
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCutVarXYAngles(const MCHTrackConv& mchTrack,
                                      const MFTTrack& mftTrack)
{

  if (mChargeCutEnabled && (mchTrack.getCharge() != mftTrack.getCharge()))
    return false;

  auto dx = mchTrack.getX() - mftTrack.getX();
  auto dy = mchTrack.getY() - mftTrack.getY();
  auto dPhi = mchTrack.getPhi() - mftTrack.getPhi();
  auto dTheta =
    TMath::Abs(EtaToTheta(mchTrack.getEta()) - EtaToTheta(mftTrack.getEta()));
  auto distance = TMath::Sqrt(dx * dx + dy * dy);
  auto cutDistance =
    3 * 1.3 * TMath::Sqrt(mchTrack.getSigma2X() + mchTrack.getSigma2Y());
  auto cutPhi = 3 * 1.15 * TMath::Sqrt(mchTrack.getSigma2Phi());
  auto cutTanl = 3 * 1.22 * TMath::Sqrt(mchTrack.getSigma2Tanl());
  return (distance < cutDistance) and (dPhi < cutPhi) and (dTheta < cutTanl);
}

//_________________________________________________________________________________________________
bool MUONMatcher::matchCutDisabled(const MCHTrackConv& mchTrack,
                                   const MFTTrack& mftTrack)
{
  return true;
}

//_________________________________________________________________________________________________
void MUONMatcher::setCutFunction(
  bool (MUONMatcher::*func)(const MCHTrackConv&, const MFTTrack&))
{
  mCutFunc = func;
  std::size_t npars = 0;
  // Setting default parameters
  if (func == &MUONMatcher::matchCutDistance)
    npars = 1;
  if (func == &MUONMatcher::matchCutDistanceSigma)
    npars = 1;
  if (func == &MUONMatcher::matchCutDistanceAndAngles)
    npars = 3;
  for (auto par = mCutParams.size(); par < npars; par++)
    setCutParam(par, 1.0);
}

//_________________________________________________________________________________________________
void MUONMatcher::finalize()
{ // compute labels and populates mMatchingHelper
  auto GTrackID = 0;
  auto nFakes = 0;
  auto nNoMatch = 0;
  auto nCloseMatches = 0;
  auto MCHTrackID = -1;
  auto nMCHTracks = 0;

  std::cout << "Computing Track Labels..." << std::endl;
  for (int event = 0; event < mNEvents; event++) {
    if (!mMatchSaveAll) {
      GTrackID = 0;
      nMCHTracks += (int)mSortedGlobalMuonTracks[event].size();
      for (auto& gTrack : mSortedGlobalMuonTracks[event]) {
        if (gTrack.closeMatch())
          nCloseMatches++;
        auto bestMFTTrackMatchID = gTrack.getBestMFTTrackMatchID();
        if (mVerbose) {
          std::cout << " GlobalTrack # " << GTrackID
                    << " chi^2 = " << gTrack.getMatchingChi2() << std::endl;
          std::cout << "    bestMFTTrackMatchID :  " << bestMFTTrackMatchID
                    << std::endl;
        }
        MCHTrackID = GTrackID;
        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(MCHTrackID);
        o2::MCCompLabel thisLabel{MCHlabel[0].getTrackID(),
                                  MCHlabel[0].getEventID(), -1, true};
        if (bestMFTTrackMatchID >= 0) {
          auto MFTlabel = mftTrackLabels.at(bestMFTTrackMatchID);
          if (mVerbose) {
            std::cout << "    MFT Label:  ";
            MFTlabel.print();
            std::cout << "    MCH Label:  ";
            MCHlabel[0].print();
          }

          if ((MCHlabel[0].getTrackID() == MFTlabel.getTrackID()) and
              (MCHlabel[0].getEventID() == MFTlabel.getEventID())) {
            thisLabel = MCHlabel[0];
            thisLabel.setFakeFlag(false);
            gTrack.computeResiduals2Cov(mMFTTracks[bestMFTTrackMatchID]);

          } else {
            nFakes++;
          }
        } else {
          nNoMatch++;
        }
        if (mVerbose) {
          std::cout << "    Global Track Label => ";
          thisLabel.print();
          std::cout << std::endl;
        }
        mSortedGlobalTrackLabels[event].addElement(mSortedGlobalTrackLabels[event].getIndexedSize(),
                                                   thisLabel);
        GTrackID++;
      }
    } else { // if matchSaveAll is set
      GTrackID = 0;
      nMCHTracks += (int)mSortedGlobalMuonTracksExt[event].size();
      for (auto& gTrack : mSortedGlobalMuonTracksExt[event]) {
        if (gTrack.closeMatch())
          nCloseMatches++;
        auto bestMFTTrackMatchID = gTrack.getBestMFTTrackMatchID();
        if (mVerbose) {
          std::cout << " GlobalTrack # " << GTrackID
                    << " chi^2 = " << gTrack.getMatchingChi2() << std::endl;
          std::cout << "    bestMFTTrackMatchID :  " << bestMFTTrackMatchID
                    << std::endl;
        }
        MCHTrackID = gTrack.getMCHTrackID();
        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(MCHTrackID);
        o2::MCCompLabel thisLabel{MCHlabel[0].getTrackID(),
                                  MCHlabel[0].getEventID(), -1, true};
        if (bestMFTTrackMatchID >= 0) {
          auto MFTlabel = mftTrackLabels.at(bestMFTTrackMatchID);
          if (mVerbose) {
            std::cout << "    MFT Label:  ";
            MFTlabel.print();
            std::cout << "    MCH Label:  ";
            MCHlabel[0].print();
          }

          if ((MCHlabel[0].getTrackID() == MFTlabel.getTrackID()) and
              (MCHlabel[0].getEventID() == MFTlabel.getEventID())) {
            thisLabel = MCHlabel[0];
            thisLabel.setFakeFlag(false);
            gTrack.computeResiduals2Cov(mMFTTracks[bestMFTTrackMatchID]);

          } else {
            nFakes++;
          }
        } else {
          nNoMatch++;
        }
        if (mVerbose) {
          std::cout << "    Global Track Label => ";
          thisLabel.print();
          std::cout << std::endl;
        }
        mSortedGlobalTrackLabels[event].addElement(mSortedGlobalTrackLabels[event].getIndexedSize(),
                                                   thisLabel);

        GTrackID++;
      }
    }
  }
  auto nCorrectMatches = nMCHTracks - nFakes - nNoMatch;

  MatchingHelper& helper = mMatchingHelper;
  helper.nMCHTracks = nMCHTracks;
  helper.nCloseMatches = nCloseMatches;
  helper.nCorrectMatches = nCorrectMatches;
  helper.nFakes = nFakes;
  helper.nNoMatch = nNoMatch;

  std::cout << "********************************** Matching Summary "
               "********************************** "
            << std::endl;

  if (mMatchSaveAll) {
    std::cout << "*WARNING* Summary is invalid for option --matchSaveAll !!! " << std::endl;
  }

  std::cout << helper.nMCHTracks << " MCH Tracks in " << mNEvents << " events" << std::endl;
  std::cout << helper.nNoMatch << " dangling MCH tracks ("
            << 100.0 * nNoMatch / nMCHTracks << "%)" << std::endl;
  std::cout << helper.nGMTracks() << " global muon tracks (efficiency = "
            << 100.0 * helper.getPairingEfficiency() << "%)" << std::endl;
  std::cout << helper.nCorrectMatches
            << " Correct Match GM tracks (Correct Match Ratio = "
            << 100.0 * helper.getCorrectMatchRatio() << "%)" << std::endl;
  std::cout << nCloseMatches
            << " Close matches - correct MFT match in search window"
            << " (" << 100. * nCloseMatches / (helper.nGMTracks()) << "%)"
            << std::endl;
  std::cout << helper.nFakes << " Fake GM tracks (contamination = "
            << 100.0 * (1.0 - helper.getCorrectMatchRatio()) << ")"
            << std::endl;
  std::cout << "***************************************************************"
               "*********************** "
            << std::endl;
  std::cout << " Annotation: " << helper.Annotation() << std::endl;
  std::cout << "***************************************************************"
               "*********************** "
            << std::endl;
}

//_________________________________________________________________________________________________
void MUONMatcher::saveGlobalMuonTracks()
{
  std::vector<GlobalMuonTrack>* tracks;
  std::vector<GlobalMuonTrackExt>* tracksExt;
  MCLabels* trackLabels;
  std::string gmtracksfilename;
  gmtracksfilename = mMatchSaveAll ? "GlobalMuonTracksExt.root" : "GlobalMuonTracks.root";
  TFile outFile(gmtracksfilename.c_str(), "RECREATE");
  TTree outTree("o2sim", "Global Muon Tracks");
  outTree.Branch("GlobalMuonTrackMCTruth", &trackLabels);

  if (!mMatchSaveAll) {
    outTree.Branch("GlobalMuonTrack", &tracks);
  } else {
    outTree.Branch("GlobalMuonTrackExt", &tracksExt);
  }

  for (int event = 0; event < mNEvents; event++) {
    tracks = &mSortedGlobalMuonTracks[event];
    tracksExt = &mSortedGlobalMuonTracksExt[event];
    trackLabels = &mSortedGlobalTrackLabels[event];

    std::cout << " Saving " << tracks->size() << "  GlobalMuonTracks for event  " << event << " of " << mNEvents << std::endl;

    outTree.Fill();
  }
  outTree.Write();
  outFile.WriteObjectAny(&mMatchingHelper, "MatchingHelper", "Matching Helper");
  outFile.Close();
  std::cout << "Global Muon Tracks saved to " << gmtracksfilename << std::endl;

  std::ofstream matcherConfig("MatchingConfig.txt");
  matcherConfig << mMatchingHelper.MatchingConfig() << std::endl;
  matcherConfig.close();
}

//_________________________________________________________________________________________________
void MUONMatcher::fitTracks()
{
  std::cout << "Fitting global muon tracks..." << std::endl;

  auto GTrackID = 0;

  for (auto& globalMuonTracks : mSortedGlobalMuonTracks) {
    for (auto& gTrack : globalMuonTracks) {
      if (gTrack.getBestMFTTrackMatchID() >= 0) {
        if (mVerbose)
          std::cout << "  ==> Fitting Global Track # " << GTrackID
                    << " with MFT track # " << gTrack.getBestMFTTrackMatchID()
                    << ":" << std::endl;
        fitGlobalMuonTrack(gTrack);
      } else {
        if (mVerbose)
          std::cout << "No matching candidate for MCH Track " << GTrackID
                    << std::endl;
      }
      GTrackID++;
    }
  }
  GTrackID = 0;
  for (auto& globalMuonTracksExt : mSortedGlobalMuonTracksExt) {
    for (auto& gTrack : globalMuonTracksExt) {
      if (gTrack.getBestMFTTrackMatchID() >= 0) {
        if (mVerbose)
          std::cout << "  ==> Fitting Global Track Ext # " << GTrackID
                    << " with MFT track # " << gTrack.getBestMFTTrackMatchID()
                    << ":" << std::endl;
        fitGlobalMuonTrack(gTrack);
      } else {
        if (mVerbose)
          std::cout << "No matching candidate for MCH Track " << GTrackID
                    << std::endl;
      }
      GTrackID++;
    }
  }
  std::cout << "Finished fitting global muon tracks." << std::endl;
}

//_________________________________________________________________________________________________
void MUONMatcher::fitGlobalMuonTrack(GlobalMuonTrack& gTrack)
{

  const auto& mftTrack = mMFTTracks[gTrack.getBestMFTTrackMatchID()];
    
  
  /*
  if(mMatchingPlaneZ<-90.0){
    MFTTrack new_mftTrack = MFTTrack();
    float slopeX = gTrack.getPx()/gTrack.getPz();
    float slopeY = gTrack.getPy()/gTrack.getPz();
    float bendP  = sqrt(gTrack.getPy()*gTrack.getPy() + gTrack.getPz()*gTrack.getPz());

    MCHTrack* mchTrack = new MCHTrack();
    mchTrack->setZ(gTrack.getZ());
    mchTrack->setNonBendingCoor(gTrack.getX());
    mchTrack->setNonBendingSlope(slopeX);
    mchTrack->setBendingCoor(gTrack.getY());
    mchTrack->setBendingSlope(slopeY);
    mchTrack->setInverseBendingMomentum(1./bendP);
    mchTrack->setCharge(gTrack.getCharge());

    mMCHTrackExtrap.extrapToMatchingPlane(mchTrack, 0);

    // Parameter conversion
    double alpha1, alpha3, alpha4, x2, x3, x4;

    alpha1 = mchTrack->getNonBendingSlope();
    alpha3 = mchTrack->getBendingSlope();
    alpha4 = mchTrack->getInverseBendingMomentum();

    x2 = TMath::ATan2(-alpha3, -alpha1);
    x3 = -1. / TMath::Sqrt(alpha3 * alpha3 + alpha1 * alpha1);
    x4 = alpha4 * -x3 * TMath::Sqrt(1 + alpha3 * alpha3);
    // Set output
    gTrack.setX(mchTrack->getNonBendingCoor());
    gTrack.setY(mchTrack->getBendingCoor());
    gTrack.setZ(mchTrack->getZ());
    gTrack.setPhi(x2);
    gTrack.setTanl(x3);
    gTrack.setInvQPt(x4);
    gTrack.setCharge(mchTrack->getCharge());
  
    
    new_mftTrack.setParameters(mftTrack.getOutParam().getParameters());
    new_mftTrack.setCovariances(mftTrack.getOutParam().getCovariances());
    new_mftTrack.setZ(mftTrack.getOutParam().getZ());
    new_mftTrack.propagateToZhelix(0, mField_z);
    
    new_mftTrack.setNumberOfPoints(mftTrack.getNumberOfPoints());
    new_mftTrack.setExternalClusterIndexOffset(mftTrack.getExternalClusterIndexOffset());

    auto ncls = new_mftTrack.getNumberOfPoints();
    auto offset = new_mftTrack.getExternalClusterIndexOffset();
    auto invQPt0 = gTrack.getInvQPt();
    auto sigmainvQPtsq = gTrack.getCovariances()(4, 4);
  
    // initialize the starting track parameters and cluster
    auto nPoints = new_mftTrack.getNumberOfPoints();
    auto k = TMath::Abs(o2::constants::math::B2C * mField_z);
    auto Hz = std::copysign(1, mField_z);
    if (mVerbose) {

      std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
      std::cout << "  N Clusters = " << ncls << std::endl;
      std::cout << "  Best MFT Track Match ID " << gTrack.getBestMFTTrackMatchID()
		<< std::endl;
      std::cout << "  MCHTrack: X = " << gTrack.getX() << " Y = " << gTrack.getY()
		<< " Z = " << gTrack.getZ() << " Tgl = " << gTrack.getTanl()
		<< "  Phi = " << gTrack.getPhi() << " pz = " << gTrack.getPz()
		<< " qpt = " << 1.0 / gTrack.getInvQPt() << std::endl;
    }

    gTrack.setX(new_mftTrack.getX());
    gTrack.setY(new_mftTrack.getY());
    gTrack.setZ(new_mftTrack.getZ());
    gTrack.setPhi(new_mftTrack.getPhi());
    gTrack.setTanl(new_mftTrack.getTanl());
    gTrack.setInvQPt(gTrack.getInvQPt());
  
    if (mVerbose) {

      std::cout << "  MFTTrack: X = " << new_mftTrack.getX()
		<< " Y = " << new_mftTrack.getY() << " Z = " << new_mftTrack.getZ()
		<< " Tgl = " << new_mftTrack.getTanl()
		<< "  Phi = " << new_mftTrack.getPhi() << " pz = " << new_mftTrack.getPz()
		<< " qpt = " << 1.0 / new_mftTrack.getInvQPt() << std::endl;
      std::cout << "  initTrack GlobalTrack: q/pt = " << gTrack.getInvQPt() << std::endl;
    }

    SMatrix55Sym lastParamCov;
    Double_t tanlsigma = TMath::Max(std::abs(new_mftTrack.getTanl()), .5);
    Double_t qptsigma = TMath::Max(std::abs(new_mftTrack.getInvQPt()), .5);
    
    lastParamCov(0, 0) = 10000. * new_mftTrack.getCovariances()(0, 0); // <X,X>
    lastParamCov(1, 1) = 10000. * new_mftTrack.getCovariances()(1, 1); // <Y,X>
    lastParamCov(2, 2) = 10000. * new_mftTrack.getCovariances()(2, 2); // TMath::Pi() * TMath::Pi() / 16 // <PHI,X>
    lastParamCov(3, 3) = 10000. * new_mftTrack.getCovariances()(3, 3); // 100. * tanlsigma * tanlsigma;  // new_mftTrack.getCovariances()(3, 3);     // <TANL,X>
    lastParamCov(4, 4) = gTrack.getCovariances()(4, 4);            //100. * qptsigma * qptsigma;  // <INVQPT,X>

    gTrack.setCovariances(lastParamCov);
  
    for (int icls = ncls - 1; icls > -1; --icls) {
      auto clsEntry = mtrackExtClsIDs[offset + icls];
      auto& thiscluster = mMFTClusters[clsEntry];
      computeCluster(gTrack, thiscluster);
    } 
  }
  else{
  */
    auto ncls = mftTrack.getNumberOfPoints();
    auto offset = mftTrack.getExternalClusterIndexOffset();
    auto invQPt0 = gTrack.getInvQPt();
    auto sigmainvQPtsq = gTrack.getCovariances()(4, 4);
  
    // initialize the starting track parameters and cluster
    auto nPoints = mftTrack.getNumberOfPoints();
    auto k = TMath::Abs(o2::constants::math::B2C * mField_z);
    auto Hz = std::copysign(1, mField_z);
    if (mVerbose) {

      std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
      std::cout << "  N Clusters = " << ncls << std::endl;
      std::cout << "  Best MFT Track Match ID " << gTrack.getBestMFTTrackMatchID()
		<< std::endl;
      std::cout << "  MCHTrack: X = " << gTrack.getX() << " Y = " << gTrack.getY()
		<< " Z = " << gTrack.getZ() << " Tgl = " << gTrack.getTanl()
		<< "  Phi = " << gTrack.getPhi() << " pz = " << gTrack.getPz()
		<< " qpt = " << 1.0 / gTrack.getInvQPt() << std::endl;
    }

    gTrack.setX(mftTrack.getX());
    gTrack.setY(mftTrack.getY());
    gTrack.setZ(mftTrack.getZ());
    gTrack.setPhi(mftTrack.getPhi());
    gTrack.setTanl(mftTrack.getTanl());
    gTrack.setInvQPt(gTrack.getInvQPt());
  
    if (mVerbose) {

      std::cout << "  MFTTrack: X = " << mftTrack.getX()
		<< " Y = " << mftTrack.getY() << " Z = " << mftTrack.getZ()
		<< " Tgl = " << mftTrack.getTanl()
		<< "  Phi = " << mftTrack.getPhi() << " pz = " << mftTrack.getPz()
		<< " qpt = " << 1.0 / mftTrack.getInvQPt() << std::endl;
      std::cout << "  initTrack GlobalTrack: q/pt = " << gTrack.getInvQPt() << std::endl;
    }

    SMatrix55Sym lastParamCov;
    Double_t tanlsigma = TMath::Max(std::abs(mftTrack.getTanl()), .5);
    Double_t qptsigma = TMath::Max(std::abs(mftTrack.getInvQPt()), .5);

    lastParamCov(0, 0) = 10000. * mftTrack.getCovariances()(0, 0); // <X,X>
    lastParamCov(1, 1) = 10000. * mftTrack.getCovariances()(1, 1); // <Y,X>
    lastParamCov(2, 2) = 10000. * mftTrack.getCovariances()(2, 2); // TMath::Pi() * TMath::Pi() / 16 // <PHI,X>
    lastParamCov(3, 3) = 10000. * mftTrack.getCovariances()(3, 3); // 100. * tanlsigma * tanlsigma;  // mftTrack.getCovariances()(3, 3);     // <TANL,X>
    lastParamCov(4, 4) = gTrack.getCovariances()(4, 4);            //100. * qptsigma * qptsigma;  // <INVQPT,X>

    gTrack.setCovariances(lastParamCov);
  
    for (int icls = ncls - 1; icls > -1; --icls) {
      auto clsEntry = mtrackExtClsIDs[offset + icls];
      auto& thiscluster = mMFTClusters[clsEntry];
      computeCluster(gTrack, thiscluster);
    }
    //  }

}

//_________________________________________________________________________________________________
bool MUONMatcher::computeCluster(GlobalMuonTrack& track, MFTCluster& cluster)
{
  /// Propagate track to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  const auto& clx = cluster.getX();
  const auto& cly = cluster.getY();
  const auto& clz = cluster.getZ();

  if (mVerbose) {
    std::cout << "computeCluster:     X = " << clx << " Y = " << cly << " Z = " << clz << " nCluster = " << cluster << std::endl;
  }

  if (!propagateToNextClusterWithMCS(track, clz)) {
    return false;
  }

  if (mVerbose) {
    std::cout << "   AfterExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
    std::cout << "Track covariances after extrap: \n"
              << track.getCovariances() << std::endl
              << std::endl;
  }

  // recompute parameters
  const std::array<float, 2>& pos = {clx, cly};
  const std::array<float, 2>& cov = {cluster.sigmaX2, cluster.sigmaY2};

  if (track.update(pos, cov)) {
    if (mVerbose) {
      std::cout << "   New Cluster: X = " << clx << " Y = " << cly << " Z = " << clz << std::endl;
      std::cout << "   AfterKalman: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;
      std::cout << std::endl;
      std::cout << "Track covariances after Kalman update: \n"
                << track.getCovariances() << std::endl
                << std::endl;
    }
    return true;
  }
  return false;
}

//_________________________________________________________________________________________________
GlobalMuonTrack MUONMatcher::MCHtoGlobal(MCHTrack& mchTrack)
{
  // Convert a MCH Track parameters and covariances matrix to the
  // GlobalMuonTrack format. Must be called after propagation on the absorber

  using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;

  GlobalMuonTrack convertedTrack;

  // Parameter conversion
  double alpha1, alpha3, alpha4, x2, x3, x4;

  alpha1 = mchTrack.getNonBendingSlope();
  alpha3 = mchTrack.getBendingSlope();
  alpha4 = mchTrack.getInverseBendingMomentum();

  x2 = TMath::ATan2(-alpha3, -alpha1);
  x3 = -1. / TMath::Sqrt(alpha3 * alpha3 + alpha1 * alpha1);
  x4 = alpha4 * -x3 * TMath::Sqrt(1 + alpha3 * alpha3);

  auto K = alpha1 * alpha1 + alpha3 * alpha3;
  auto K32 = K * TMath::Sqrt(K);
  auto L = TMath::Sqrt(alpha3 * alpha3 + 1);

  // Covariances matrix conversion
  SMatrix55Std jacobian;
  SMatrix55Sym covariances;

  if (mVerbose) {

    std::cout << " MCHtoGlobal - MCH Covariances:\n";
    std::cout << " mchTrack.getCovariances()(0, 0) =  "
              << mchTrack.getCovariances()(0, 0)
              << " ; mchTrack.getCovariances()(2, 2) = "
              << mchTrack.getCovariances()(2, 2) << std::endl;
  }
  covariances(0, 0) = mchTrack.getCovariances()(0, 0);
  covariances(0, 1) = mchTrack.getCovariances()(0, 1);
  covariances(0, 2) = mchTrack.getCovariances()(0, 2);
  covariances(0, 3) = mchTrack.getCovariances()(0, 3);
  covariances(0, 4) = mchTrack.getCovariances()(0, 4);

  covariances(1, 1) = mchTrack.getCovariances()(1, 1);
  covariances(1, 2) = mchTrack.getCovariances()(1, 2);
  covariances(1, 3) = mchTrack.getCovariances()(1, 3);
  covariances(1, 4) = mchTrack.getCovariances()(1, 4);

  covariances(2, 2) = mchTrack.getCovariances()(2, 2);
  covariances(2, 3) = mchTrack.getCovariances()(2, 3);
  covariances(2, 4) = mchTrack.getCovariances()(2, 4);

  covariances(3, 3) = mchTrack.getCovariances()(3, 3);
  covariances(3, 4) = mchTrack.getCovariances()(3, 4);

  covariances(4, 4) = mchTrack.getCovariances()(4, 4);

  jacobian(0, 0) = 1;

  jacobian(1, 2) = 1;

  jacobian(2, 1) = -alpha3 / K;
  jacobian(2, 3) = alpha1 / K;

  jacobian(3, 1) = alpha1 / K32;
  jacobian(3, 3) = alpha3 / K32;

  jacobian(4, 1) = -alpha1 * alpha4 * L / K32;
  jacobian(4, 3) = alpha3 * alpha4 * (1 / (TMath::Sqrt(K) * L) - L / K32);
  jacobian(4, 4) = L / TMath::Sqrt(K);

  // jacobian*covariances*jacobian^T
  covariances = ROOT::Math::Similarity(jacobian, covariances);

  // Set output
  convertedTrack.setX(mchTrack.getNonBendingCoor());
  convertedTrack.setY(mchTrack.getBendingCoor());
  convertedTrack.setZ(mchTrack.getZ());
  convertedTrack.setPhi(x2);
  convertedTrack.setTanl(x3);
  convertedTrack.setInvQPt(x4);
  convertedTrack.setCharge(mchTrack.getCharge());
  convertedTrack.setCovariances(covariances);

  return convertedTrack;
}

//_________________________________________________________________________________________________
double MUONMatcher::matchMFT_MCH_TracksXY(const MCHTrackConv& mchTrack,
                                          const MFTTrack& mftTrack)
{
  // Calculate Matching Chi2 - X and Y positions

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
  SMatrix25 H_k;
  SMatrix22 V_k;
  SVector2 m_k(mftTrack.getX(), mftTrack.getY()), r_k_kminus1;
  SMatrix5 GlobalMuonTrackParameters = mchTrack.getParameters();
  SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
  V_k(0, 0) = mftTrack.getCovariances()(0, 0);
  V_k(1, 1) = mftTrack.getCovariances()(1, 1);
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;

  // Covariance of residuals
  SMatrix22 invResCov =
    (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix52 K_k =
    GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 =
    m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction
  // GlobalMuonTrackParameters = GlobalMuonTrackParameters + K_k * r_k_kminus1;

  // Update covariances Matrix
  // SMatrix55Std updatedCov = (I - K_k * H_k) * GlobalMuonTrackCovariances;

  auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

  // GlobalMuonTrack matchTrack(mchTrack);
  // matchTrack.setZ(mchTrack.getZ());
  // matchTrack.setParameters(GlobalMuonTrackParameters);
  // matchTrack.setCovariances(GlobalMuonTrackCovariances);
  // matchTrack.setMatchingChi2(matchChi2Track);
  return matchChi2Track;
}

//_________________________________________________________________________________________________
double
  MUONMatcher::matchMFT_MCH_TracksXYPhiTanl(const MCHTrackConv& mchTrack,
                                            const MFTTrack& mftTrack)
{
  // Match two tracks evaluating positions & angles

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
  SMatrix45 H_k;
  SMatrix44 V_k;
  SVector4 m_k(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
               mftTrack.getTanl()),
    r_k_kminus1;
  SMatrix5 GlobalMuonTrackParameters = mchTrack.getParameters();
  SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
  V_k(0, 0) = mftTrack.getCovariances()(0, 0);
  V_k(1, 1) = mftTrack.getCovariances()(1, 1);
  V_k(2, 2) = mftTrack.getCovariances()(2, 2);
  V_k(3, 3) = mftTrack.getCovariances()(3, 3);
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;
  H_k(2, 2) = 1.0;
  H_k(3, 3) = 1.0;

  // Covariance of residuals
  SMatrix44 invResCov =
    (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix54 K_k =
    GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 =
    m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction
  // GlobalMuonTrackParameters = GlobalMuonTrackParameters + K_k * r_k_kminus1;

  // Update covariances Matrix
  // SMatrix55Std updatedCov = (I - K_k * H_k) * GlobalMuonTrackCovariances;

  auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

  // GlobalMuonTrack matchTrack(mchTrack);
  // matchTrack.setZ(mchTrack.getZ());
  // matchTrack.setParameters(GlobalMuonTrackParameters);
  // matchTrack.setCovariances(GlobalMuonTrackCovariances);
  // matchTrack.setMatchingChi2(matchChi2Track);
  return matchChi2Track;
}

//_________________________________________________________________________________________________
double MUONMatcher::matchMFT_MCH_TracksAllParam(const MCHTrackConv& mchTrack,
                                                const MFTTrack& mftTrack)
{
  // Match two tracks evaluating all parameters: X,Y, phi, tanl & q/pt

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity(), H_k, V_k;
  SVector5 m_k(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
               mftTrack.getTanl(), mftTrack.getInvQPt()),
    r_k_kminus1;
  SMatrix5 GlobalMuonTrackParameters = mchTrack.getParameters();
  SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
  V_k(0, 0) = mftTrack.getCovariances()(0, 0);
  V_k(1, 1) = mftTrack.getCovariances()(1, 1);
  V_k(2, 2) = mftTrack.getCovariances()(2, 2);
  V_k(3, 3) = mftTrack.getCovariances()(3, 3);
  V_k(4, 4) = mftTrack.getCovariances()(4, 4);
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;
  H_k(2, 2) = 1.0;
  H_k(3, 3) = 1.0;
  H_k(4, 4) = 1.0;

  // Covariance of residuals
  SMatrix55Std invResCov =
    (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix55Std K_k =
    GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 =
    m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction
  // GlobalMuonTrackParameters = GlobalMuonTrackParameters + K_k * r_k_kminus1;

  // Update covariances Matrix
  // SMatrix55Std updatedCov = (I - K_k * H_k) * GlobalMuonTrackCovariances;

  auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

  // GlobalMuonTrack matchTrack(mchTrack);
  // matchTrack.setZ(mchTrack.getZ());
  // matchTrack.setParameters(GlobalMuonTrackParameters);
  // matchTrack.setCovariances(GlobalMuonTrackCovariances);
  // matchTrack.setMatchingChi2(matchChi2Track);
  return matchChi2Track;
}

//_________________________________________________________________________________________________
double MUONMatcher::matchHiroshima(const GlobalMuonTrack& mchTrack,
                                   const MFTTrack& mftTrack)
{

  //Hiroshima's Matching function

  //Matching constants
  Double_t LAbs = 415.;    //Absorber Length[cm]
  Double_t mumass = 0.106; //mass of muon [GeV/c^2]
  Double_t l;              //the length that extrapolated MCHtrack passes through absorber

  if (mMatchingPlaneZ >= -90.0) {
    l = LAbs;
  } else {
    l = 505.0 + mMatchingPlaneZ;
  }

  //defference between MFTtrack and MCHtrack

  auto dx = mftTrack.getX() - mchTrack.getX();
  auto dy = mftTrack.getY() - mchTrack.getY();
  auto dthetax = TMath::ATan(mftTrack.getPx() / TMath::Abs(mftTrack.getPz())) - TMath::ATan(mchTrack.getPx() / TMath::Abs(mchTrack.getPz()));
  auto dthetay = TMath::ATan(mftTrack.getPy() / TMath::Abs(mftTrack.getPz())) - TMath::ATan(mchTrack.getPy() / TMath::Abs(mchTrack.getPz()));

  //Multiple Scattering(=MS)

  auto pMCH = mchTrack.getP();
  auto lorentzbeta = pMCH / TMath::Sqrt(mumass * mumass + pMCH * pMCH);
  auto zMS = copysign(1.0, mchTrack.getCharge());
  auto thetaMS = 13.6 / (1000.0 * pMCH * lorentzbeta * 1.0) * zMS * TMath::Sqrt(60.0 * l / LAbs) * (1.0 + 0.038 * TMath::Log(60.0 * l / LAbs));
  auto xMS = thetaMS * l / TMath::Sqrt(3.0);

  //normalize by theoritical Multiple Coulomb Scattering width to be momentum-independent
  //make the dx and dtheta dimensionless

  auto dxnorm = dx / xMS;
  auto dynorm = dy / xMS;
  auto dthetaxnorm = dthetax / thetaMS;
  auto dthetaynorm = dthetay / thetaMS;

  //rotate distribution

  auto dxrot = dxnorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaxnorm * TMath::Sin(TMath::Pi() / 4.0);
  auto dthetaxrot = dxnorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaxnorm * TMath::Cos(TMath::Pi() / 4.0);
  auto dyrot = dynorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaynorm * TMath::Sin(TMath::Pi() / 4.0);
  auto dthetayrot = dynorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaynorm * TMath::Cos(TMath::Pi() / 4.0);

  //convert ellipse to circle

  auto k = 0.7; //need to optimize!!
  auto dxcircle = dxrot;
  auto dycircle = dyrot;
  auto dthetaxcircle = dthetaxrot / k;
  auto dthetaycircle = dthetayrot / k;

  //score

  auto scoreX = TMath::Sqrt(dxcircle * dxcircle + dthetaxcircle * dthetaxcircle);
  auto scoreY = TMath::Sqrt(dycircle * dycircle + dthetaycircle * dthetaycircle);
  auto score = TMath::Sqrt(scoreX * scoreX + scoreY * scoreY);

  return score;
};

//_________________________________________________________________________________________________
double MUONMatcher::matchHiroshimaReverse(const GlobalMuonTrack& mchTrack,
                                   const MFTTrack& mftTrack)
{

  //Hiroshima's Matching function

  //Matching constants
  Double_t LAbs = 415.;    //Absorber Length[cm]
  Double_t mumass = 0.106; //mass of muon [GeV/c^2]
  Double_t l;              //the length that extrapolated MCHtrack passes through absorber

  if (-505.0<=mMatchingPlaneZ&&mMatchingPlaneZ<-90.0) {
    l = 90.0 - mMatchingPlaneZ;
  } else if (mMatchingPlaneZ<-505.0){
    l=LAbs;
  }

  //defference between MFTtrack and MCHtrack

  auto dx = -mftTrack.getX() + mchTrack.getX();
  auto dy = -mftTrack.getY() + mchTrack.getY();
  auto dthetax = -TMath::ATan(mftTrack.getPx() / TMath::Abs(mftTrack.getPz())) + TMath::ATan(mchTrack.getPx() / TMath::Abs(mchTrack.getPz()));
  auto dthetay = -TMath::ATan(mftTrack.getPy() / TMath::Abs(mftTrack.getPz())) + TMath::ATan(mchTrack.getPy() / TMath::Abs(mchTrack.getPz()));

  //Multiple Scattering(=MS)

  auto pMFT = mchTrack.getP(); //Use MCHtrack's momentum as MFTtrack's momentum
  auto lorentzbeta = pMFT / TMath::Sqrt(mumass * mumass + pMFT * pMFT);
  auto zMS = copysign(1.0, mftTrack.getCharge());
  auto thetaMS = 13.6 / (1000.0 * pMFT * lorentzbeta * 1.0) * zMS * TMath::Sqrt(60.0 * l / LAbs) * (1.0 + 0.038 * TMath::Log(60.0 * l / LAbs));
  auto xMS = thetaMS * l / TMath::Sqrt(3.0);

  //normalize by theoritical Multiple Coulomb Scattering width to be momentum-independent
  //make the dx and dtheta dimensionless

  auto dxnorm = dx / xMS;
  auto dynorm = dy / xMS;
  auto dthetaxnorm = dthetax / thetaMS;
  auto dthetaynorm = dthetay / thetaMS;

  //rotate distribution

  auto dxrot = dxnorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaxnorm * TMath::Sin(TMath::Pi() / 4.0);
  auto dthetaxrot = dxnorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaxnorm * TMath::Cos(TMath::Pi() / 4.0);
  auto dyrot = dynorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaynorm * TMath::Sin(TMath::Pi() / 4.0);
  auto dthetayrot = dynorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaynorm * TMath::Cos(TMath::Pi() / 4.0);

  //convert ellipse to circle

  auto k = 2.4; //need to optimize!!
  auto dxcircle = dxrot;
  auto dycircle = dyrot;
  auto dthetaxcircle = dthetaxrot / k;
  auto dthetaycircle = dthetayrot / k;

  //score

  auto scoreX = TMath::Sqrt(dxcircle * dxcircle + dthetaxcircle * dthetaxcircle);
  auto scoreY = TMath::Sqrt(dycircle * dycircle + dthetaycircle * dthetaycircle);
  auto score = TMath::Sqrt(scoreX * scoreX + scoreY * scoreY);

  return score;
};

//_________________________________________________________________________________________________
double MUONMatcher::matchTrainedML(const MCHTrackConv& mchTrack,
                                   const MFTTrack& mftTrack)
{

  setMLFeatures(mchTrack, mftTrack);
  double matchingscore;
  matchingscore = mTMVAReader->EvaluateRegression(0, "MUONMatcherML");

  //  Note: returning negative ML scores to get lowest value = best match
  return -matchingscore;
}

//_________________________________________________________________________________________________
void MUONMatcher::EvaluateML()
{
  std::cout << " Checking ML performance... " << std::endl;

  TFile* outputfile = new TFile("ML_Evaluation.root", "RECREATE");

  int npoints = 25;
  float cutsteps = 1. / (float)npoints;
  TGraph* gr1 = new TGraph(npoints);
  gr1->SetMarkerColor(kGreen + 2);
  gr1->SetMarkerStyle(20);
  TGraph* gr2 = new TGraph(npoints);
  gr2->SetMarkerColor(2);
  gr2->SetMarkerStyle(20);
  TGraph* gr3 = new TGraph(npoints);
  gr3->SetMarkerColor(4);
  gr3->SetMarkerStyle(20);
  TGraph* gr4 = new TGraph(npoints);
  gr4->SetMarkerColor(9);
  gr4->SetMarkerStyle(20);

  auto GTrackID = 0;
  int correct_match, fake, reject, nMCHtracks = mMatchingHelper.nMCHTracks;

  for (float cut = 0.05; cut < 1.; cut += cutsteps) {
    correct_match = fake = reject = 0;
    for (int event = 0; event < mNEvents; event++) {
      GTrackID = 0;
      for (auto& gTrack : mSortedGlobalMuonTracks[event]) {
        auto GMTracklabel = mSortedGlobalTrackLabels[event].getLabels(GTrackID);
        auto bestMFTTrackMatchID = gTrack.getBestMFTTrackMatchID();
        if (bestMFTTrackMatchID >= 0 && std::abs(gTrack.getMatchingChi2()) > cut)
          GMTracklabel[0].isCorrect() ? (correct_match++) : (fake++);
        else
          reject++;
        GTrackID++;
      } //loop over global tracks
    }

    gr1->SetPoint(gr1->GetN(), cut, (float)correct_match / (nMCHtracks - reject));
    gr2->SetPoint(gr2->GetN(), cut, (float)fake / (nMCHtracks - reject));
    gr3->SetPoint(gr3->GetN(), cut, (float)reject / nMCHtracks);
    gr4->SetPoint(gr4->GetN(), (float)reject / nMCHtracks, (float)correct_match / (nMCHtracks - reject));
  }

  TCanvas* c1 = new TCanvas("c1", "c1");
  gr1->SetTitle("Correct Match Ratio Vs Score Cut");
  gr1->Draw("ap");
  gr1->GetXaxis()->SetTitle("Score Cut");
  gr1->GetYaxis()->SetTitle("Correct Match Ratio");
  c1->SetName("Correct Match Ratio");
  c1->Write();
  c1->Close();
  TCanvas* c2 = new TCanvas("c2", "c2");
  gr2->SetTitle("Fake Match Ratio Vs Score Cut");
  gr2->Draw("ap");
  gr2->GetXaxis()->SetTitle("Score Cut");
  gr2->GetYaxis()->SetTitle("Fake Match Ratio");
  c2->SetName("Fake Match Ratio");
  c2->Write();
  c2->Close();
  TCanvas* c3 = new TCanvas("c3", "c3");
  gr3->SetTitle("Tracks Rejected by Score Cut");
  gr3->Draw("ap");
  gr3->GetXaxis()->SetTitle("Score Cut");
  gr3->GetYaxis()->SetTitle("MCH Tracks Rejection Ratio");
  c3->SetName("Reject Tracks ratio");
  c3->Write();
  c3->Close();
  TCanvas* c123 = new TCanvas("c123", "c123");
  gr1->Draw("ap");
  gr2->Draw("p same");
  gr3->Draw("p same");
  c123->SetTitle("all ratios");
  c123->SetName("All Ratios");
  c123->Write();
  c123->Close();
  TCanvas* c4 = new TCanvas("c4", "c4");
  gr4->SetTitle("Correct Match Ratio Vs Rejection Ratio");
  gr4->Draw("ap");
  gr4->GetXaxis()->SetTitle("MCH Tracks Rejection Ratio");
  gr4->GetYaxis()->SetTitle("Correct Match ratio");
  c4->SetName("Correct & Reject");
  c4->Write();
  c4->Close();

  //Creating Histograms
  TCanvas* score_canvas = new TCanvas("scores hist", "c0");
  TH1D* correctmatch_hist = new TH1D("Correct Matches", "Correct Matches Scores", 50, -0.1, 1.15);
  correctmatch_hist->SetLineColor(kGreen + 1);
  correctmatch_hist->SetLineWidth(5);
  correctmatch_hist->SetXTitle("Matching Score");
  TH1D* fakematch_hist = new TH1D("Fake Matches", "Fake Matches Scores", 50, -0.1, 1.15);
  fakematch_hist->SetLineColor(2);
  fakematch_hist->SetLineWidth(5);
  fakematch_hist->SetXTitle("Matching Score");

  for (int event = 0; event < mNEvents; event++) {
    GTrackID = 0;
    for (auto& gTrack : mSortedGlobalMuonTracks[event]) {

      auto mMCHTrackID = gTrack.getMCHTrackID();
      auto GMTracklabel = mSortedGlobalTrackLabels[event].getLabels(GTrackID);
      auto bestMFTTrackMatchID = gTrack.getBestMFTTrackMatchID();
      auto bestchi2 = std::abs(gTrack.getMatchingChi2());

      if (bestMFTTrackMatchID >= 0) {
        if (GMTracklabel[0].isCorrect()) {
          correctmatch_hist->Fill(bestchi2);
        } else {
          fakematch_hist->Fill(bestchi2);
        }
      }
      if (bestchi2 < mMLScoreCut) { // TMVA: Drop MFT candidate if score bellow threshold
        gTrack.setBestMFTTrackMatchID(-1);
        mSortedGlobalTrackLabels[event].getLabels(GTrackID)[0].setFakeFlag(true);
      }
      GTrackID++;
    } //end loop over global tracks
  }   // end loop over events
  correctmatch_hist->SetStats(0);
  correctmatch_hist->Draw();
  fakematch_hist->SetStats(0);
  fakematch_hist->Draw("SAME");
  score_canvas->Write();
  // --- Write histograms and graphics
  outputfile->Write();
  outputfile->Close();
}

//_________________________________________________________________________________________________
void MUONMatcher::exportTrainingDataRoot(int nMCHTracks)
{
  mMatchingHelper.MatchingFunction = "";
  if (nMCHTracks < 0 or nMCHTracks > mMatchingHelper.nMCHTracks)
    nMCHTracks = mMatchingHelper.nMCHTracks;

  std::string outputfile("MLTraining_" + std::to_string(nMCHTracks) + "_MCHTracks.root");

  auto fT = TFile::Open(outputfile.c_str(), "RECREATE");
  std::cout << " Exporting training data to TTree. Pairing MFT tracks with " << nMCHTracks << " MCH Tracks" << std::endl;

  Int_t Truth, track_IDs, nCorrectPairs = 0, nFakesPairs = 0;
  Int_t pairID = 0;
  TTree* matchTree = new TTree("matchTree", "MatchTree");
  for (std::size_t nFeature = 0; nFeature < mNInputFeatures; nFeature++) {
    matchTree->Branch(mMLInputFeaturesName[nFeature].c_str(), &mMLInputFeatures[nFeature], Form("%s/F", mMLInputFeaturesName[nFeature].c_str()));
  }

  matchTree->Branch("Truth", &Truth, "Truth/I");

  auto event = 0;
  while (nMCHTracks > 0 or event < mNEvents) {
    for (auto& mchTracks : mSortedGlobalMuonTracks) {
      if (mVerbose) {
        std::cout << " Event #" << event << std::endl;
        std::cout << "  MCHTracks in this event = " << mSortedGlobalMuonTracks[event].size() << std::endl;
        std::cout << "  nMFTTracks = " << mSortedMFTTracks[event].size() << std::endl;
      }
      auto MCHTrackID = 0;
      for (auto& mchTrack : mchTracks) {
        if (!nMCHTracks)
          continue;
        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(MCHTrackID);
        if (mVerbose) {
          std::cout << "  MCHTrack #" << MCHTrackID << " (" << nMCHTracks << " left)" << std::endl;
        }
        auto mftTrackID = 0;
        for (auto mftTrack : mSortedMFTTracks[event]) {
          auto MFTlabel = mftTrackLabels.at(mftTrackLabelsIDx[event][mftTrackID]);
          Truth = (int)(MFTlabel.getTrackID() == MCHlabel[0].getTrackID());
          
	  if ((mCorrectMatchIgnoreCut && Truth) || matchingCut(mchTrack, mftTrack)) {
	    if( (Truth == 1) || ( Truth==0 && gRandom->Rndm() < 1 )){
	      setMLFeatures(mchTrack, mftTrack);	    
	      Truth ? nCorrectPairs++ : nFakesPairs++;
	      pairID++;
	      matchTree->Fill();
	    }
          }
          mftTrackID++;
        } // loop mfttracks
        nMCHTracks--;
        MCHTrackID++;
      }
      event++;
    }
  }

  fT->Write();
  auto nPairs = nCorrectPairs + nFakesPairs;
  std::cout << "Exported training data: " << nPairs << " pairs (" << nCorrectPairs << " correct pairs ; " << nFakesPairs << " fake pairs)" << std::endl;
  std::cout << "   Exported file name: " << outputfile << std::endl;

  std::ofstream matcherConfig("MatchingConfig.txt");
  matcherConfig << mMatchingHelper.MatchingConfig() << std::endl;
  matcherConfig.close();
}

//_________________________________________________________________________________________________
void MUONMatcher::runONNXRuntime()
{
  
  std::cout<<"ONNXRUntime is running....."<<endl;

  std::string model_file = "";

  model_file = gSystem->Getenv("ML_MODULE");

  std::cout << " Using " << model_file << std::endl;

  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  Ort::Experimental::Session session = Ort::Experimental::Session(env, model_file, session_options);  // access experimental components via the Experimental namespace

  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> output_shapes;

  input_names = session.GetInputNames();
  input_shapes = session.GetInputShapes();
  output_names = session.GetOutputNames();
  output_shapes = session.GetOutputShapes();

  auto input_shape = input_shapes[0];
  input_shape[0] = 1;

  // Assume model has 1 input node and 1 output node.
  //assert(input_names.size() == 1 && output_names.size() == 1);
  
  // Runs matching over all tracks on a single event
  std::cout << "Number of MCH Tracks = " << mMatchingHelper.nMCHTracks << std::endl;
  std::cout << "Annotation: " << mMatchingHelper.Annotation() << std::endl;
  std::cout << "Running runEventMatching for " << mNEvents << " events"
            << std::endl;

  uint32_t GTrackID = 0;
  auto nMCHTracks = 0;
  auto nGlobalMuonTracksExt = 0;
  for (int event = 0; event < mNEvents; event++) {

    nMCHTracks += (int)mSortedGlobalMuonTracks[event].size();
    if (mVerbose)
      std::cout << "Matching event # " << event << ": " << mSortedGlobalMuonTracks[event].size() << " MCH Tracks" << std::endl;
    GTrackID = 0;

    if (!mMatchSaveAll) {
      for (auto& gTrack : mSortedGlobalMuonTracks[event]) {
        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(GTrackID);
	gTrack.setMatchingChi2(-999);
          auto mftTrackID = 0;
          for (auto mftTrack : mSortedMFTTracks[event]) {
            
	    auto MFTlabel = mftTrackLabels.at(mftTrackLabelsIDx[event][mftTrackID]);
            
	    if (matchingCut(gTrack, mftTrack)) {

	      gTrack.countCandidate();

	      if (MFTlabel.getTrackID() == MCHlabel[0].getTrackID()){
                gTrack.setCloseMatch();
	      }

	      std::vector<float> input_tensor_values = getTrainingVariables(gTrack,mftTrack);
	      std::vector<Ort::Value> input_tensors;
	      input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
	      std::vector<Ort::Value> output_tensors = session.Run(input_names, input_tensors, output_names);

	      const int* output_label = output_tensors[0].GetTensorData<int>();
	      
	      Float_t prob = -999;
	      
	      if(model_file.find("lightGBM") != std::string::npos){
		const float* output_value = output_tensors[1].GetTensorData<float>();
		if( output_value[0] <  output_value[1])
		  prob = output_value[1];
	      }
	      
	      else if(model_file.find("tfNN") != std::string::npos){
		const float* output_value = output_tensors[0].GetTensorData<float>();
		prob = output_value[0];
	      }

	      if (prob > gTrack.getMatchingChi2()) {
		gTrack.setBestMFTTrackMatchID(mftTrackLabelsIDx[event][mftTrackID]);
		gTrack.setMatchingChi2(prob);
	      }

            }
            mftTrackID++;
          }
        GTrackID++;
      }      // /loop over global tracks
    } else { // if matchSaveAll is set
      // store all given matches for performance studies
      // loop MCH tracks
      for (auto& track : mSortedMCHTracks[event]) {

        mMCHTrackExtrap.extrapToVertexWithoutBranson(&track, mMatchingPlaneZ);
        GlobalMuonTrackExt gTrackTmp{MCHtoGlobal(track)};

        auto MCHlabel = mSortedMCHTrackLabels[event].getLabels(GTrackID);
        auto mftTrackID = 0;

        for (auto mftTrack : mSortedMFTTracks[event]) {

	  auto MFTlabel = mftTrackLabels.at(mftTrackLabelsIDx[event][mftTrackID]);

	  if (MFTlabel.getEventID() == event) {
            if (matchingCut(gTrackTmp, mftTrack)) {

	      GlobalMuonTrackExt gTrack{MCHtoGlobal(track)};
              gTrack.setParametersMCH(gTrack.getParameters());
              gTrack.setCovariancesMCH(gTrack.getCovariances());
              gTrack.countCandidate();

	      if (MFTlabel.getTrackID() == MCHlabel[0].getTrackID()){
                gTrack.setCloseMatch();
              }

              auto chi2 = matchingEval(gTrack, mftTrack);
              gTrack.setBestMFTTrackMatchID(mftTrackLabelsIDx[event][mftTrackID]);
              gTrack.setParametersMFT(mftTrack.getParameters());
              gTrack.setCovariancesMFT(mftTrack.getCovariances());
              gTrack.setMCHTrackID(GTrackID);

	      std::vector<float> input_tensor_values = getTrainingVariables(gTrack,mftTrack);

	      std::vector<Ort::Value> input_tensors;
	      input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));

	      std::vector<Ort::Value> output_tensors = session.Run(input_names, input_tensors, output_names);

	      //assert(output_tensors.size() == session.GetOutputNames().size() && output_tensors[0].IsTensor());
	      
	      //cout<<"HHHHH         "<<output_tensors.size()<<"   "<<session.GetOutputNames().size()<<endl;
	      
	      const int* output_label = output_tensors[0].GetTensorData<int>();

	      Float_t prob = -999;

	      if(model_file.find("lightGBM") != std::string::npos){
		const float* output_value = output_tensors[1].GetTensorData<float>();
		prob = output_value[1];
	      }

	      else if(model_file.find("tfNN") != std::string::npos){
		const float* output_value = output_tensors[0].GetTensorData<float>();
		prob = output_value[0];
	      }
	      
	      gTrack.setMatchingChi2(prob);
	      
	      mSortedGlobalMuonTracksExt[event].push_back(gTrack);
            } // matching cut
          }   // end event match, MFT
          mftTrackID++;
        } // end loop MFT tracks
        GTrackID++;
      } // end loop MCH tracks
    }   // end match save all
    nGlobalMuonTracksExt += (int)mSortedGlobalMuonTracksExt[event].size();
  }     // /loop over events

  if (!mMatchSaveAll) {
    std::cout << "Finished runEventMatching on " << nMCHTracks << " MCH Tracks on " << mNEvents << " events"
              << std::endl;
  } else {
    std::cout << "Finished runEventMatching with matchSaveAll option " << nGlobalMuonTracksExt << " global muon tracks."
              << std::endl;
  }
  finalize();
  if (mTMVAReader)
    EvaluateML();

}

std::vector<float> MUONMatcher::getTrainingVariables(const MCHTrackConv& mchTrack,const MFTTrack& mftTrack)
{

  Float_t MFT_X      = mftTrack.getX();
  Float_t MFT_Y      = mftTrack.getY();
  Float_t MFT_Phi    = mftTrack.getPhi();
  Float_t MFT_Tanl   = mftTrack.getTanl();
  Float_t MFT_Eta    = mftTrack.getEta();
  Float_t MFT_InvQPt = mftTrack.getInvQPt();
  Float_t MFT_QPt    = mftTrack.getPt()*mftTrack.getCharge();
  Float_t MFT_Pt     = 1./abs(MFT_QPt);
  Float_t MFT_Px     = TMath::Cos(MFT_Phi) * MFT_Pt;
  Float_t MFT_Py     = TMath::Sin(MFT_Phi) * MFT_Pt;
  Float_t MFT_Pz     = MFT_Tanl*MFT_Pt;
  Float_t MFT_P      = MFT_Pt * sqrt(1+MFT_Tanl*MFT_Tanl);
  Float_t MFT_Ch     = MFT_QPt>0 ? +1 : -1;
  Float_t MFT_Cov00  = mftTrack.getCovariances()(0, 0);
  Float_t MFT_Cov01  = mftTrack.getCovariances()(0, 1);
  Float_t MFT_Cov11  = mftTrack.getCovariances()(1, 1);
  Float_t MFT_Cov02  = mftTrack.getCovariances()(0, 2);
  Float_t MFT_Cov12  = mftTrack.getCovariances()(1, 2);
  Float_t MFT_Cov22  = mftTrack.getCovariances()(2, 2);
  Float_t MFT_Cov03  = mftTrack.getCovariances()(0, 3);
  Float_t MFT_Cov13  = mftTrack.getCovariances()(1, 3);
  Float_t MFT_Cov23  = mftTrack.getCovariances()(2, 3);
  Float_t MFT_Cov33  = mftTrack.getCovariances()(3, 3);
  Float_t MFT_Cov04  = mftTrack.getCovariances()(0, 4);
  Float_t MFT_Cov14  = mftTrack.getCovariances()(1, 4);
  Float_t MFT_Cov24  = mftTrack.getCovariances()(2, 4);
  Float_t MFT_Cov34  = mftTrack.getCovariances()(3, 4);
  Float_t MFT_Cov44  = mftTrack.getCovariances()(4, 4);

  Float_t MCH_X      = mchTrack.getX();
  Float_t MCH_Y      = mchTrack.getY();
  Float_t MCH_Phi    = mchTrack.getPhi();
  Float_t MCH_Tanl   = mchTrack.getTanl();
  Float_t MCH_Eta    = mchTrack.getEta();
  Float_t MCH_InvQPt = mchTrack.getInvQPt();
  Float_t MCH_QPt    = mchTrack.getPt()*mchTrack.getCharge();
  Float_t MCH_Pt     = 1./abs(MCH_QPt);
  Float_t MCH_Px     = TMath::Cos(MCH_Phi) * MCH_Pt;
  Float_t MCH_Py     = TMath::Sin(MCH_Phi) * MCH_Pt;
  Float_t MCH_Pz     = MCH_Tanl*MCH_Pt;
  Float_t MCH_P      = MCH_Pt * sqrt(1+MCH_Tanl*MCH_Tanl);
  Float_t MCH_Ch     = MCH_QPt>0 ? +1 : -1;
  Float_t MCH_Cov00  = mchTrack.getCovariances()(0, 0);
  Float_t MCH_Cov01  = mchTrack.getCovariances()(0, 1);
  Float_t MCH_Cov11  = mchTrack.getCovariances()(1, 1);
  Float_t MCH_Cov02  = mchTrack.getCovariances()(0, 2);
  Float_t MCH_Cov12  = mchTrack.getCovariances()(1, 2);
  Float_t MCH_Cov22  = mchTrack.getCovariances()(2, 2);
  Float_t MCH_Cov03  = mchTrack.getCovariances()(0, 3);
  Float_t MCH_Cov13  = mchTrack.getCovariances()(1, 3);
  Float_t MCH_Cov23  = mchTrack.getCovariances()(2, 3);
  Float_t MCH_Cov33  = mchTrack.getCovariances()(3, 3);
  Float_t MCH_Cov04  = mchTrack.getCovariances()(0, 4);
  Float_t MCH_Cov14  = mchTrack.getCovariances()(1, 4);
  Float_t MCH_Cov24  = mchTrack.getCovariances()(2, 4);
  Float_t MCH_Cov34  = mchTrack.getCovariances()(3, 4);
  Float_t MCH_Cov44  = mchTrack.getCovariances()(4, 4);
	      
  Float_t MFT_TrackChi2   = mftTrack.getTrackChi2();
  Float_t MFT_NClust      = mftTrack.getNumberOfPoints();
	      
  Float_t Delta_X      = MCH_X      - MFT_X;
  Float_t Delta_Y      = MCH_Y      - MFT_Y;
  Float_t Delta_XY     = sqrt(pow(MCH_X-MFT_X,2) + pow(MCH_Y-MFT_Y,2));
  Float_t Delta_Phi    = MCH_Phi    - MFT_Phi;
  Float_t Delta_Tanl   = MCH_Tanl   - MFT_Tanl;
  Float_t Delta_Eta    = MCH_Eta    - MFT_Eta;
  Float_t Delta_InvQPt = MCH_InvQPt - MFT_InvQPt;
  Float_t Delta_Pt     = MCH_Pt     - MFT_Pt;
  Float_t Delta_Px     = MCH_Px     - MFT_Px;
  Float_t Delta_Py     = MCH_Py     - MFT_Py;
  Float_t Delta_Pz     = MCH_Pz     - MFT_Pz;
  Float_t Delta_P      = MCH_P      - MFT_P ;
  Float_t Delta_Ch     = MCH_Ch     - MFT_Ch ;

  Float_t Ratio_X      = MCH_X      / MFT_X;
  Float_t Ratio_Y      = MCH_Y      / MFT_Y;
  Float_t Ratio_Phi    = MCH_Phi    / MFT_Phi;
  Float_t Ratio_Tanl   = MCH_Tanl   / MFT_Tanl;
  Float_t Ratio_Eta    = MCH_Eta    / MFT_Eta;
  Float_t Ratio_InvQPt = MCH_InvQPt / MFT_InvQPt;
  Float_t Ratio_Pt     = MCH_Pt     / MFT_Pt;
  Float_t Ratio_Px     = MCH_Px     / MFT_Px;
  Float_t Ratio_Py     = MCH_Py     / MFT_Py;
  Float_t Ratio_Pz     = MCH_Pz     / MFT_Pz;
  Float_t Ratio_P      = MCH_P      / MFT_P ;
  Float_t Ratio_Ch     = MCH_Ch     / MFT_Ch;

  Float_t MFT_TrackReducedChi2 = MFT_TrackChi2/MFT_NClust;	      
  Float_t MatchingScore = matchingEval(mchTrack, mftTrack);
  
  std::vector<float> input_tensor_values{
      MFT_InvQPt,
      MFT_X,
      MFT_Y,
      MFT_Phi,
      MFT_Tanl,

      MCH_InvQPt,
      MCH_X,
      MCH_Y,
      MCH_Phi,
      MCH_Tanl,

      MFT_TrackChi2,
      MFT_NClust,
      MFT_TrackReducedChi2,

      Delta_InvQPt,
      Delta_X,
      Delta_Y,
      Delta_XY,
      Delta_Phi,
      Delta_Tanl,

      Delta_Ch
  };


  return input_tensor_values;



}

//_________________________________________________________________________________________________
Float_t EtaToTheta(Float_t arg)
{
  return (180. / TMath::Pi()) * 2. * atan(exp(-arg));
}
