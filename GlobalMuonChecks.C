#if !defined(__CLING__) || defined(__ROOTCLING__)

#ifdef __MAKECINT__
#pragma link C++ class GlobalMuonTrack + ;
#pragma link C++ class std::vector < GlobalMuonTrack> + ;
#pragma link C++ class MatchingHelper + ;
#endif

#include "CommonConstants/MathConstants.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TEfficiency.h"
#include "TFile.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TTree.h"
#include <TGeoGlobalMagField.h>
#include <TGraph.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TMath.h>
#include <TProfile.h>
#include <TStyle.h>

#endif

#include "include/GlobalMuonTrack.h"
#include "macrohelpers/HistosHelpers.C"
#include "macrohelpers/MagField.C"

using o2::MCTrackT;
using GlobalMuonTrack = o2::track::GlobalMuonTrack;
using eventFoundTracks = std::vector<bool>;
using std::vector;
vector<eventFoundTracks> allFoundGMTracks; // True for reconstructed tracks -
                                           // one vector of bool per event

bool DEBUG_VERBOSE = false;
bool EXPORT_HISTOS_IMAGES = false;

//_________________________________________________________________________________________________
int GlobalMuonChecks(const std::string trkFile = "GlobalMuonTracks.root",
		     const std::string perfecttrkFile = "perfectGlobalMuonTracks.root",
                     const std::string o2sim_KineFile = "o2sim_Kine.root")
{

  if (gSystem->Getenv("VERBOSEMATCHING")) {
    std::cout << " Vebose checking enabled." << std::endl;
    DEBUG_VERBOSE = true;
  }

  // Histos parameters
  Double_t pMin = 0.0;
  Double_t pMax = 100.0;
  Double_t deltaetaMin = -.1;
  Double_t deltaetaMax = +.1;
  Double_t etaMin = -3.5;
  Double_t etaMax = -2.4;
  Double_t deltaphiMin = -.2; //-3.15,
  Double_t deltaphiMax = .2;  //+3.15,
  Double_t deltatanlMin = -2.0;
  Double_t deltatanlMax = 2.0;

  // histos
  // gROOT->SetStyle("Bold");
  gStyle->SetOptStat("emr");
  gStyle->SetStatW(.28);
  gStyle->SetStatH(.26);
  gStyle->SetPalette(1, 0);
  gStyle->SetCanvasColor(10);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetFrameLineWidth(3);
  gStyle->SetFrameFillColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  // gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetHistLineWidth(2);
  gStyle->SetHistLineColor(kRed);
  gStyle->SetFuncWidth(2);
  gStyle->SetFuncColor(kGreen);
  gStyle->SetLineWidth(2);
  gStyle->SetLabelSize(0.06, "xyz");
  gStyle->SetLabelOffset(0.01, "y");
  // gStyle->SetLabelColor(kBlue,"xy");
  gStyle->SetTitleSize(0.06, "xyz");
  gStyle->SetTitleSize(0.08, "o");
  gStyle->SetTitleOffset(0.95, "Y");
  gStyle->SetTitleFillColor(10);
  // gStyle->SetTitleTextColor(kNlacBlue);
  gStyle->SetStatColor(10);

  enum TH2HistosCodes {
    kGMTrackDeltaXYVertex,
    kGMTrackDeltaXYVertex0_1,
    kGMTrackDeltaXYVertex1_4,
    kGMTrackDeltaXYVertex4plus,
    kGMTrackChi2vsFitChi2,
    kGMTrackQPRec_MC,
    kGMTrackPtResolution,
    kGMTrackInvPtResolution,
    kMCTracksEtaZ
  };

  std::map<int, const char*> TH2Names{
    {kGMTrackDeltaXYVertex, "Global Muon Tracks Vertex at Z = 0"},
    {kGMTrackDeltaXYVertex0_1, "Global Muon Tracks Vertex at Z = 0 Pt0_1"},
    {kGMTrackDeltaXYVertex1_4, "Global Muon Tracks Vertex at Z = 0 Pt1_4"},
    {kGMTrackDeltaXYVertex4plus,
     "Global Muon Tracks Vertex at Z = 0 Pt4plus"},
    {kGMTrackChi2vsFitChi2, "Global Muon TracksChi2vsFitChi2"},
    {kGMTrackQPRec_MC, "GM Track QP FITxMC"},
    {kGMTrackPtResolution, "GM Track Pt Resolution"},
    {kGMTrackInvPtResolution, "GM Track InvPt Resolution"},
    {kMCTracksEtaZ, "MCTracks_eta_z"}};

  std::map<int, const char*> TH2Titles{
    {kGMTrackDeltaXYVertex, "Global Muon Tracks at Z_vertex"},
    {kGMTrackDeltaXYVertex0_1, "Global Muon Tracks at Z_vertex (pt < 1)"},
    {kGMTrackDeltaXYVertex1_4, "Global Muon Tracks at Z_vertex (1 < pt < 4)"},
    {kGMTrackDeltaXYVertex4plus, "Global Muon Tracks at Z_vertex (pt > 4)"},
    {kGMTrackChi2vsFitChi2, "Tracks Chi2 vs FitChi2"},
    {kGMTrackQPRec_MC, "Charged Momentum: Reconstructed vs MC"},
    {kGMTrackPtResolution, "Pt Resolution"},
    {kGMTrackInvPtResolution, "InvPt Resolution"},
    {kMCTracksEtaZ, "MC Tracks: Pseudorapidity vs zVertex"}};

  std::map<int, std::array<double, 6>> TH2Binning{
    {kGMTrackDeltaXYVertex, {100, -.5, .5, 100, -.5, .5}},
    {kGMTrackDeltaXYVertex0_1, {100, -.5, .5, 100, -.5, .5}},
    {kGMTrackDeltaXYVertex1_4, {100, -.5, .5, 100, -.5, .5}},
    {kGMTrackDeltaXYVertex4plus, {100, -.5, .5, 100, -.5, .5}},
    {kGMTrackChi2vsFitChi2, {500, 0, 1000, 250, 0., 500.}},
    {kGMTrackQPRec_MC, {50, -100, 100, 50, -100, 100}},
    {kGMTrackPtResolution, {20, 0, 10, 100, 0, 5}},
    {kGMTrackInvPtResolution, {14, 0, 7, 300, -2, 2}},
    {kMCTracksEtaZ, {31, -15, 16, 25, etaMin, etaMax}}};

  std::map<int, const char*> TH2XaxisTitles{
    {kGMTrackDeltaXYVertex, "\\Delta x ~[mm]"},
    {kGMTrackDeltaXYVertex0_1, "\\Delta x ~[mm]"},
    {kGMTrackDeltaXYVertex1_4, "\\Delta x ~[mm]"},
    {kGMTrackDeltaXYVertex4plus, "\\Delta x ~[mm]"},
    {kGMTrackChi2vsFitChi2, "Fit ~ \\chi^2"},
    {kGMTrackQPRec_MC, "(q.p)_{MC} [GeV]"},
    {kGMTrackPtResolution, "pt_{MC} [GeV]"},
    {kGMTrackInvPtResolution, "pt_{MC} [GeV]"},
    {kMCTracksEtaZ, "Vertex PosZ [cm]"}};

  std::map<int, const char*> TH2YaxisTitles{
    {kGMTrackDeltaXYVertex, "\\Delta y ~[mm]"},
    {kGMTrackDeltaXYVertex0_1, "\\Delta y ~[mm]"},
    {kGMTrackDeltaXYVertex1_4, "\\Delta y ~[mm]"},
    {kGMTrackDeltaXYVertex4plus, "\\Delta y ~[mm]"},
    {kGMTrackChi2vsFitChi2, "Track ~ \\chi^2"},
    {kGMTrackQPRec_MC, "(q.p)_{fit} [GeV]"},
    {kGMTrackPtResolution, "pt_{fit} / pt_{MC}"},
    {kGMTrackInvPtResolution, "(1/(p_t)_{fit} - 1/(p_t)_{MC})*(p_t)_{MC}"},
    {kMCTracksEtaZ, "\\eta"}};

  enum TH1HistosCodes {
    kGMTrackDeltaXErr,
    kGMTrackDeltaYErr,
    kGMTrackDeltaPhiErr,
    kGMTrackDeltaTanLErr,
    kGMTrackDeltainvQPtErr,
    kMCHResTrackDeltaXErr,
    kMCHResTrackDeltaYErr,
    kMCHResTrackDeltaPhiErr,
    kMCHResTrackDeltaTanLErr,
    kMCHResTrackDeltainvQPtErr,
    kGMTrackXChi2,
    kGMTrackYChi2,
    kGMTrackPhiChi2,
    kGMTrackTanlChi2,
    kGMTrackinvQPtChi2,
    kFitChi2,
    kGMTracksP,
    kGMTrackDeltaTanl,
    kGMTrackDeltaTanl0_1,
    kGMTrackDeltaTanl1_4,
    kGMTrackDeltaTanl4plus,
    kGMTrackDeltaPhi,
    kGMTrackDeltaPhi0_1,
    kGMTrackDeltaPhi1_4,
    kGMTrackDeltaPhi4plus,
    kGMTrackDeltaPhiDeg,
    kGMTrackDeltaPhiDeg0_1,
    kGMTrackDeltaPhiDeg1_4,
    kGMTrackDeltaPhiDeg4plus,
    kGMTrackDeltaInvQPt,
    kGMTrackDeltaX,
    kGMTrackDeltaX0_1,
    kGMTrackDeltaX1_4,
    kGMTrackDeltaX4plus,
    kGMTrackDeltaY,
    kGMTrackR,
    kGMTrackQ,
    kGMTrackQ0_1,
    kGMTrackQ1_4,
    kGMTrackQ4plus,
    kGMTrackChi2,
    kMCTrackspT,
    kMCTracksp,
    kMCTrackEta
  };

  std::map<int, const char*> TH1Names{
    {kGMTracksP, "Global Muon Tracks Fitted p"},
    {kGMTrackDeltaXErr, "Delta X / SigmaX"},
    {kGMTrackDeltaYErr, "Delta Y / SigmaY"},
    {kGMTrackDeltaPhiErr, "Delta Phi at Vertex / SigmaPhi"},
    {kGMTrackDeltaTanLErr, "Delta_Tanl / SigmaTanl"},
    {kGMTrackDeltainvQPtErr, "Delta_InvQPt / Sigma_{q/pt}"},
    {kMCHResTrackDeltaXErr, "MCH Delta X / SigmaX"},
    {kMCHResTrackDeltaYErr, "MCH Delta Y / SigmaY"},
    {kMCHResTrackDeltaPhiErr, "MCH Delta Phi at Vertex / SigmaPhi"},
    {kMCHResTrackDeltaTanLErr, "MCH Delta_Tanl / SigmaTanl"},
    {kMCHResTrackDeltainvQPtErr, "MCH Delta_InvQPt / Sigma_{q/pt}"},
    {kGMTrackDeltaTanl, "Global Muon Tracks Fitted Delta_tanl"},
    {kGMTrackXChi2, "X Chi2"},
    {kGMTrackYChi2, "Y Chi2"},
    {kGMTrackPhiChi2, "Phi chi2"},
    {kGMTrackTanlChi2, "Tanl Chi2"},
    {kGMTrackinvQPtChi2, "InvQPt Chi2"},
    {kFitChi2, "Fit Chi2"},
    {kGMTrackDeltaTanl0_1, "Global Muon Tracks tanl (pt < 1)"},
    {kGMTrackDeltaTanl1_4, "Global Muon Tracks tanl (1 < pt < 4)"},
    {kGMTrackDeltaTanl4plus, "Global Muon Tracks tanl (pt > 4)"},
    {kGMTrackDeltaPhi, "Global Muon Tracks Fitted Phi at Vertex"},
    {kGMTrackDeltaPhi0_1,
     "Global Muon Tracks Fitted Phi at Vertex [rad] (pt < 1)"},
    {kGMTrackDeltaPhi1_4,
     "Global Muon Tracks Fitted Phi at Vertex [rad] (1 < pt < 4)"},
    {kGMTrackDeltaPhi4plus,
     "Global Muon Tracks Fitted Phi at Vertex [rad] (pt > 4)"},
    {kGMTrackDeltaPhiDeg, "Global Muon Tracks Fitted Phi at Vertex [deg]"},
    {kGMTrackDeltaPhiDeg0_1,
     "Global Muon Tracks Fitted Phi at Vertex [deg] (pt < 1)"},
    {kGMTrackDeltaPhiDeg1_4,
     "Global Muon Tracks Fitted Phi at Vertex [deg] (1 < pt < 4)"},
    {kGMTrackDeltaPhiDeg4plus,
     "Global Muon Tracks Fitted Phi at Vertex [deg] (pt > 4)"},
    {kGMTrackDeltaInvQPt, "Global Muon Tracks invQPt"},
    {kGMTrackDeltaX, "Global Muon Tracks Delta X"},
    {kGMTrackDeltaX0_1, "Global Muon Tracks Delta X (pt < 1)"},
    {kGMTrackDeltaX1_4, "Global Muon Tracks Delta X (1 < pt < 4)"},
    {kGMTrackDeltaX4plus, "Global Muon Tracks Delta X (pt > 4)"},
    {kGMTrackDeltaY, "Global Muon Tracks Delta Y"},
    {kGMTrackR, "Global Muon Tracks Delta R"},
    {kGMTrackQ, "Charge Match"},
    {kGMTrackQ0_1, "Charge Match (pt < 1)"},
    {kGMTrackQ1_4, "Charge Match (1 < pt < 4)"},
    {kGMTrackQ4plus, "Charge Match (pt > 4)"},
    {kGMTrackChi2, "Tracks Chi2"},
    {kMCTrackspT, "MC Tracks p_T"},
    {kMCTracksp, "MC Tracks p"},
    {kMCTrackEta, "MC Tracks eta"}};

  std::map<int, const char*> TH1Titles{
    {kGMTracksP, "Standalone Global Muon Tracks P"},
    {kGMTrackDeltaXErr, "\\Delta X / \\sigma_X"},
    {kGMTrackDeltaYErr, "\\Delta Y / \\sigma_Y"},
    {kGMTrackDeltaPhiErr, "\\Delta \\phi / \\sigma_\\phi"},
    {kGMTrackDeltaTanLErr, "\\Delta TanL / \\sigma_{TanL} "},
    {kGMTrackDeltainvQPtErr, "\\Delta(q/Pt) / \\sigma_{q/pt}"},
    {kMCHResTrackDeltaXErr, "\\Delta X / \\sigma_X"},
    {kMCHResTrackDeltaYErr, "\\Delta Y / \\sigma_Y"},
    {kMCHResTrackDeltaPhiErr, "\\Delta \\phi / \\sigma_\\phi"},
    {kMCHResTrackDeltaTanLErr, "\\Delta TanL / \\sigma_{TanL} "},
    {kMCHResTrackDeltainvQPtErr, "\\Delta(q/Pt) / \\sigma_{q/pt}"},
    {kGMTrackXChi2, "\\chi^2(x)"},
    {kGMTrackYChi2, "\\chi^2(y)"},
    {kGMTrackPhiChi2, "\\chi^2(\\phi)"},
    {kGMTrackTanlChi2, "\\chi^2(TanL)"},
    {kGMTrackinvQPtChi2, "\\chi^2(InvQP_t)"},
    {kFitChi2, "Fit Chi2"},
    {kGMTrackDeltaTanl, "tanl_{Fit} - tanl_{MC} "},
    {kGMTrackDeltaTanl0_1, "tanl_{Fit} - tanl_{MC} (pt < 1)"},
    {kGMTrackDeltaTanl1_4, "tanl_{Fit} - tanl_{MC} (1 < p_t < 4)"},
    {kGMTrackDeltaTanl4plus, "tanl_{Fit} - tanl_{MC} (p_t > 4)"},
    {kGMTrackDeltaPhi, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhi0_1, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhi1_4, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhi4plus, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhiDeg, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhiDeg0_1, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhiDeg1_4, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaPhiDeg4plus, "\\phi _{Fit} - \\phi_{MC}"},
    {kGMTrackDeltaInvQPt, "Global Muon Tracks \\Delta invQPt"},
    {kGMTrackDeltaX, "Global Muon Tracks Delta X at Z_vertex"},
    {kGMTrackDeltaX0_1, "Global Muon Tracks Delta X at Z_vertex"},
    {kGMTrackDeltaX1_4, "Global Muon Tracks Delta X at Z_vertex"},
    {kGMTrackDeltaX4plus, "Global Muon Tracks Delta X at Z_vertex"},
    {kGMTrackDeltaY, "Global Muon Tracks Delta Y at Z_vertex"},
    {kGMTrackR, "Global Muon Tracks Delta R at Z_vertex"},
    {kGMTrackQ, "Global Muon Tracks Charge Match"},
    {kGMTrackQ0_1, "Global Muon Tracks Charge Match (pt < 1)"},
    {kGMTrackQ1_4, "Global Muon Tracks Charge Match (1 < pt < 4)"},
    {kGMTrackQ4plus, "Global Muon Tracks Charge Match (pt > 4)"},
    {kGMTrackChi2, "Global Muon Tracks ~ \\chi^2"},
    {kMCTrackspT, "MC Tracks p_T"},
    {kMCTracksp, "MC Tracks p"},
    {kMCTrackEta, "MC Tracks Pseudorapidity"}};

  std::map<int, std::array<double, 3>> TH1Binning{
    {kGMTracksP, {500, pMin, pMax}},
    {kGMTrackDeltaXErr, {500, -10, 10}},
    {kGMTrackDeltaYErr, {500, -10, 10}},
    {kGMTrackDeltaPhiErr, {500, -10, +10}},
    {kGMTrackDeltaTanLErr, {500, -10, +10}},
    {kGMTrackDeltainvQPtErr, {500, -50, +50}},
    {kMCHResTrackDeltaXErr, {500, -10, 10}},
    {kMCHResTrackDeltaYErr, {500, -10, 10}},
    {kMCHResTrackDeltaPhiErr, {500, -10, +10}},
    {kMCHResTrackDeltaTanLErr, {500, -10, +10}},
    {kMCHResTrackDeltainvQPtErr, {500, -50, +50}},
    {kGMTrackXChi2, {500, 0, 100}},
    {kGMTrackYChi2, {500, 0, 100}},
    {kGMTrackPhiChi2, {500, 0, 100}},
    {kGMTrackTanlChi2, {500, 0, 100}},
    {kGMTrackinvQPtChi2, {500, 0, 100}},
    {kFitChi2, {500, 0, 50}},
    {kGMTrackDeltaTanl, {1000, deltatanlMin, deltatanlMax}},
    {kGMTrackDeltaTanl0_1, {1000, deltatanlMin, deltatanlMax}},
    {kGMTrackDeltaTanl1_4, {1000, deltatanlMin, deltatanlMax}},
    {kGMTrackDeltaTanl4plus, {1000, deltatanlMin, deltatanlMax}},
    {kGMTrackDeltaPhi, {1000, deltaphiMin, deltaphiMax}},
    {kGMTrackDeltaPhi0_1, {1000, deltaphiMin, deltaphiMax}},
    {kGMTrackDeltaPhi1_4, {1000, deltaphiMin, deltaphiMax}},
    {kGMTrackDeltaPhi4plus, {1000, deltaphiMin, deltaphiMax}},
    {kGMTrackDeltaPhiDeg,
     {1000, TMath::RadToDeg() * deltaphiMin,
      TMath::RadToDeg() * deltaphiMax}},
    {kGMTrackDeltaPhiDeg0_1,
     {1000, TMath::RadToDeg() * deltaphiMin,
      TMath::RadToDeg() * deltaphiMax}},
    {kGMTrackDeltaPhiDeg1_4,
     {1000, TMath::RadToDeg() * deltaphiMin,
      TMath::RadToDeg() * deltaphiMax}},
    {kGMTrackDeltaPhiDeg4plus,
     {1000, TMath::RadToDeg() * deltaphiMin,
      TMath::RadToDeg() * deltaphiMax}},
    {kGMTrackDeltaInvQPt, {1000, -10., 10.}},
    {kGMTrackDeltaX, {1000, -.5, .5}},
    {kGMTrackDeltaX0_1, {1000, -.5, .5}},
    {kGMTrackDeltaX1_4, {1000, -.5, .5}},
    {kGMTrackDeltaX4plus, {1000, -.5, .5}},
    {kGMTrackDeltaY, {1000, -.5, .5}},
    {kGMTrackR, {250, 0, 0.5}},
    {kGMTrackQ, {5, -2.1, 2.1}},
    {kGMTrackQ0_1, {5, -2.1, 2.1}},
    {kGMTrackQ1_4, {5, -2.1, 2.1}},
    {kGMTrackQ4plus, {5, -2.1, 2.1}},
    {kGMTrackChi2, {10000, 0, 1000}},
    {kMCTrackspT, {5000, 0, 50}},
    {kMCTracksp, {1000, pMin, pMax}},
    {kMCTrackEta, {1000, etaMin, etaMax}}};

  std::map<int, const char*> TH1XaxisTitles{
    {kGMTracksP, "p [GeV]"},
    {kGMTrackDeltaXErr, "\\Delta x  /\\sigma_{x}"},
    {kGMTrackDeltaYErr, "\\Delta y  /\\sigma_{y}"},
    {kGMTrackDeltaPhiErr, "\\Delta \\phi  /\\sigma_{\\phi}"},
    {kGMTrackDeltaTanLErr, "\\Delta tanl /\\sigma_{tanl}"},
    {kGMTrackDeltainvQPtErr, "\\Delta (q/p_t)/\\sigma_{q/Pt}"},
    {kMCHResTrackDeltaXErr, "\\Delta x  /\\sigma_{x}"},
    {kMCHResTrackDeltaYErr, "\\Delta y  /\\sigma_{y}"},
    {kMCHResTrackDeltaPhiErr, "\\Delta \\phi  /\\sigma_{\\phi}"},
    {kMCHResTrackDeltaTanLErr, "\\Delta tanl /\\sigma_{tanl}"},
    {kMCHResTrackDeltainvQPtErr, "\\Delta (q/p_t)/\\sigma_{q/Pt}"},
    {kGMTrackDeltaTanl, "\\Delta tanl"},
    {kGMTrackXChi2, "\\chi^2"},
    {kGMTrackYChi2, "\\chi^2"},
    {kGMTrackPhiChi2, "\\chi^2"},
    {kGMTrackTanlChi2, "\\chi^2"},
    {kGMTrackinvQPtChi2, "\\chi^2"},
    {kFitChi2, "\\chi^2"},
    {kGMTrackDeltaTanl0_1, "\\Delta tanl"},
    {kGMTrackDeltaTanl1_4, "\\Delta tanl"},
    {kGMTrackDeltaTanl4plus, "\\Delta tanl"},
    {kGMTrackDeltaPhi, "\\Delta \\phi ~[rad]"},
    {kGMTrackDeltaPhi0_1, "\\Delta \\phi ~[rad]"},
    {kGMTrackDeltaPhi1_4, "\\Delta \\phi ~[rad]"},
    {kGMTrackDeltaPhi4plus, "\\Delta \\phi ~[rad]"},
    {kGMTrackDeltaPhiDeg, "\\Delta \\phi ~[deg]"},
    {kGMTrackDeltaPhiDeg0_1, "\\Delta \\phi ~[deg]"},
    {kGMTrackDeltaPhiDeg1_4, "\\Delta \\phi ~[deg]"},
    {kGMTrackDeltaPhiDeg4plus, "\\Delta \\phi ~[deg]"},
    {kGMTrackDeltaInvQPt, "\\Delta invQPt"},
    {kGMTrackDeltaX, "\\Delta x ~[cm]"},
    {kGMTrackDeltaX0_1, "\\Delta x ~[cm]"},
    {kGMTrackDeltaX1_4, "\\Delta x ~[cm]"},
    {kGMTrackDeltaX4plus, "\\Delta x ~[cm]"},
    {kGMTrackDeltaY, "\\Delta y ~[cm]"},
    {kGMTrackR, "\\Delta r ~[cm]"},
    {kGMTrackQ, "q_{fit}-q_{MC}"},
    {kGMTrackQ0_1, "q_{fit}-q_{MC}"},
    {kGMTrackQ1_4, "q_{fit}-q_{MC}"},
    {kGMTrackQ4plus, "q_{fit}-q_{MC}"},
    {kGMTrackChi2, "\\chi^2"},
    {kMCTrackspT, "p_t [GeV]"},
    {kMCTracksp, "p [GeV]"},
    {kMCTrackEta, " \\eta"}};

  // Create histograms
  const int nTH1Histos = TH1Names.size();
  std::vector<std::unique_ptr<TH1F>> TH1Histos(nTH1Histos);
  auto nHisto = 0;
  for (auto& h : TH1Histos) {
    h = std::make_unique<TH1F>(TH1Names[nHisto], TH1Titles[nHisto],
                               (int)TH1Binning[nHisto][0],
                               TH1Binning[nHisto][1], TH1Binning[nHisto][2]);
    h->GetXaxis()->SetTitle(TH1XaxisTitles[nHisto]);
    ++nHisto;
  }

  const int nTH2Histos = TH2Names.size();
  std::vector<std::unique_ptr<TH2F>> TH2Histos(nTH2Histos);
  auto n2Histo = 0;
  for (auto& h : TH2Histos) {
    h = std::make_unique<TH2F>(TH2Names[n2Histo], TH2Titles[n2Histo],
                               (int)TH2Binning[n2Histo][0],
                               TH2Binning[n2Histo][1], TH2Binning[n2Histo][2],
                               (int)TH2Binning[n2Histo][3],
                               TH2Binning[n2Histo][4], TH2Binning[n2Histo][5]);
    // gStyle->SetLineWidth(4);
    // gROOT->ForceStyle();
    h->GetXaxis()->SetTitle(TH2XaxisTitles[n2Histo]);
    h->GetYaxis()->SetTitle(TH2YaxisTitles[n2Histo]);
    // h->GetXaxis()->SetLabelSize(0.05);
    // h->GetXaxis()->SetTitleSize(0.05);
    // h->GetYaxis()->SetLabelSize(0.06);
    // h->GetYaxis()->SetTitleSize(0.06);
    h->SetOption("COLZ");
    ++n2Histo;
  }

  // Profiles histograms
  auto PtRes_Profile = new TProfile("Pt_res_prof", "Profile of pt{fit}/pt{MC}",
                                    14, 0, 7, 0, 20, "s");
  PtRes_Profile->GetXaxis()->SetTitle("pt_{MC}");
  PtRes_Profile->GetYaxis()->SetTitle("mean(Pt_{Fit}/Pt_{MC})");

  auto DeltaX_Profile = new TProfile("DeltaX_prof", "Vertexing resolution", 14,
                                     0, 7, -10000., 10000., "s");
  DeltaX_Profile->GetXaxis()->SetTitle("pt_{MC} [GeV]");
  DeltaX_Profile->GetYaxis()->SetTitle("\\sigma_x ~[\\mu m]");

  // TEfficiency histogram
  TEfficiency* qMatchEff = new TEfficiency(
    "QMatchEff", "Charge Match;p_t [GeV];#epsilon", 20, 0, 10);
  // qMatchEff->GetPaintedHistogram()->GetXaxis()->SetLabelSize(0.06);
  // qMatchEff->GetPaintedHistogram()->GetYaxis()->SetLabelSize(0.06);
  // qMatchEff->GetPaintedHistogram()->GetXaxis()->SetTitleSize(0.06);
  // qMatchEff->GetPaintedHistogram()->GetYaxis()->SetTitleSize(0.06);

  TEfficiency* pairedMCHTracksEff = new TEfficiency(
    "PairingEff", "Paired_tracks;p_t [GeV];#epsilon", 20, 0, 10);
  TEfficiency* globalMuonCorrectMatchRatio =
    new TEfficiency("Correct_Match_Ratio",
                    " CorrectMatchRatio "
                    "(nCorrectMatches/NGlobalMuonTracks);p_t [GeV];#epsilon",
                    20, 0, 10);
  TEfficiency* globalMuonCombinedEff =
    new TEfficiency("Global_Matching_Efficiency",
                    "Global_Matching_Efficiency "
                    "(nCorrectMatches/NMCHTracks);p_t [GeV];#epsilon",
                    20, 0, 10);
  TEfficiency* closeMatchEff = new TEfficiency(
    "Close_Match_Eff", "Close Matches;p_t [GeV];#epsilon", 20, 0, 10);

  //All
  TH1F *recoGMTrackAllPt = new TH1F("recoGMTrackAllPt","all reconstructed GMtrack's p_{T};p_{T}^{reco}[GeV/c];Entry",1000,0,10);
  TH1F *perfectGMTrackAllPt = new TH1F("perfectGMTrackAllPt","all perfect GMtrack's p_{T};p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH1F *MCtrackAllPt = new TH1F("MCtrackAllPt","all MCtrack's p_{T};p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMTrackAllPtEta = new TH2F("recoGMTrackAllPtEta","all reconsturucted GMtrack's p_{T}-#eta;p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	TH2F *perfectGMTrackAllPtEta = new TH2F("perfectGMTrackAllPtEta","all perfect GMtrack's p_{T}-#eta;p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH2F *MCtrackAllPtEta = new TH2F("MCtrackAllPtEta","all MC GMtrack's p_{T}-#eta;p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	//Reco is pairable
  TH1F *recoGMtrackPt_RecoIsPairable = new TH1F("recoGMtrackPt_RecoIsPairable","reconstructed GMtrack p_{T} (Reco is Pairable);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMtrackPtEta_RecoIsPairable = new TH2F("recoGMtrackPtEta_RecoIsPairable","reconstructed GMtrack's p_{T}-#eta (Reco is Pairable);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	TH1F *perfectGMtrackPt_RecoIsPairable = new TH1F("perfectGMtrackPt_RecoIsPairable","perfect GMtrack p_{T} (Reco is Pairable);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH2F *perfectGMtrackPtEta_RecoIsPairable = new TH2F("perfectGMtrackPtEta_RecoIsPairable","perfect GMtrack's p_{T}-#eta (Reco is Pairable);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH1F *MCtrackPt_RecoIsPairable = new TH1F("MCtrackPt_RecoIsPairable","MCtrack p_{T} (Reco is Pairable);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH2F *MCtrackPtEta_RecoIsPairable = new TH2F("MCtrackPtEta_RecoIsPairable","MCtrack's p_{T}-#eta (Reco is Pairable);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	//Reco is Not pairable
  TH1F *recoGMtrackPt_RecoIsNotPairable = new TH1F("recoGMtrackPt_RecoIsNotPairable","reconstructed GMtrack's p_{T} (Reco is Not Pairable);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
  //Reco is Close
  TH1F *recoGMtrackPt_RecoIsClose = new TH1F("recoGMtrackPt_RecoIsClose","reconstructed GMtrack's p_{T} (Reco is Close);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMtrackPtEta_RecoIsClose = new TH2F("recoGMtrackPtEta_RecoIsClose","reconstructed GMtrack's p_{T}-#eta (Reco is Close);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	TH1F *perfectGMtrackPt_RecoIsClose = new TH1F("perfectGMtrackPt_RecoIsClose","perfect GMtrack's p_{T} (Reco is Close);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH2F *perfectGMtrackPtEta_RecoIsClose = new TH2F("perfectGMtrackPtEta_RecoIsClose","perfect GMtrack's p_{T}-#eta (Reco is Close);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH1F *MCtrackPt_RecoIsClose = new TH1F("MCtrackPt_RecoIsClose","MCtrack's p_{T} (Reco is Close);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *MCtrackPtEta_RecoIsClose = new TH2F("MCtrackPtEta_RecoIsClose","MCtrack's p_{T}-#eta (Reco is Close);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	//Reco is Not Close
	TH1F *recoGMtrackPt_RecoIsNotClose = new TH1F("recoGMtrackPt_RecoIsNotClose","reconstructed GMtrack's p_{T} (Reco is Not Close);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	//Perfect is Close
	TH1F *perfectGMtrackPt_PerfectIsClose = new TH1F("perfectGMtrackPt_PerfectIsClose","perfect GMtrack's p_{T} (Perfect is Close);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH2F *perfectGMtrackPtEta_PerfectIsClose = new TH2F("perfectGMtrackPtEta_PerfectIsClose","perfect GMtrack's p_{T}-#eta (Perfect is Close);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH1F *MCtrackPt_PerfectIsClose = new TH1F("MCtrackPt_PerfectIsClose","MCtrack's p_{T} (Perfrect is Close);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *MCtrackPtEta_PerfectIsClose = new TH2F("MCtrackPtEta_PerfectIsClose","MCtrack's p_{T}-#eta (Perfect is Close);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	//Perfect is Not close
	TH2F *perfectGMtrackPtEta_PerfectIsNotClose = new TH2F("perfectGMtrackPtEta_PerfectIsNotClose","perfect GMtrack's p_{T}-#eta (Perfect is Not Close);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH2F *MCtrackPtEta_PerfectIsNotClose = new TH2F("MCtrackPtEta_PerfectIsNotClose","MCtrack's p_{T}-#eta (Perfect is Not Close);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	//Correct
	TH1F *recoGMtrackPt_RecoIsCorrect = new TH1F("recoGMtrackPt_RecoIsCorrect","reconstructed GMtrack's p_{T} (Reco is Correct);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH1F *perfectGMtrackPt_RecoIsCorrect = new TH1F("perfectGMtrackPt_RecoIsCorrect","perfect GMtrack's p_{T} (Reco is Correct);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH1F *MCtrackPt_RecoIsCorrect = new TH1F("MCtrackPt_RecoIsCorrect","MCtrack's p_{T} (Reco is Correct);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMtrackPtEta_RecoIsCorrect = new TH2F("recoGMtrackPtEta_RecoIsCorrect","reconstructed GMtrack's p_{T}-#eta (Reco is Correct);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	TH2F *perfectGMtrackPtEta_RecoIsCorrect = new TH2F("perfectGMtrackPtEta_RecoIsCorrect","perfect GMtrack's p_{T}-#eta (Reco is Correct);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH2F *MCtrackPtEta_RecoIsCorrect = new TH2F("MCtrackPtEta_RecoIsCorrect","MCtrack's p_{T}-#eta (Reco is Correct);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	//Fake
	TH1F *recoGMtrackPt_RecoIsFake = new TH1F("recoGMtrackPt_RecoIsFake","reconstructed GMtrack's p_{T} (Reco is Fake);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH1F *perfectGMtrackPt_RecoIsFake = new TH1F("perfectGMtrackPt_RecoIsFake","perfect GMtrack's p_{T} (Reco is Fake);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH1F *MCtrackPt_RecoIsFake = new TH1F("MCtrackPt_RecoIsFake","MCtrack's p_{T} (Reco is Fake);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMtrackPtEta_RecoIsFake = new TH2F("recoGMtrackPtEta_RecoIsFake","reconstructed GMtrack's p_{T}-#eta (Reco is Fake);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	TH2F *perfectGMtrackPtEta_RecoIsFake = new TH2F("perfectGMtrackPtEta_RecoIsFake","perfect GMtrack's p_{T}-#eta (Reco is Fake);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH2F *MCtrackPtEta_RecoIsFake = new TH2F("MCtrackPtEta_RecoIsFake","MCtrack's p_{T}-#eta (Reco is Fake);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	TH1F *MCtrackPt_RecoIsFakeInPairable = new TH1F("MCtrackPt_RecoIsFakeInPairable","MCtrack's p_{T} (Reco is Fake, Pairable);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *MCtrackPtEta_RecoIsFakeInPairable = new TH2F("MCtrackPtEta_RecoIsFakeInPairable","MCtrack's p_{T}-#eta (Reco is Fake, Pairable);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);

	//Dangling
	TH1F *recoGMtrackPt_RecoIsDangling = new TH1F("recoGMtrackPt_RecoIsDangling","reconstructed GMtrack's p_{T} (Reco is Dangling);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH1F *perfectGMtrackPt_RecoIsDangling = new TH1F("perfectGMtrackPt_RecoIsDangling","perfect GMtrack's p_{T} (Reco is Dangling);p_{T}^{perfect}[GeV/c];Entry",1000,0,10);
	TH1F *MCtrackPt_RecoIsDangling = new TH1F("MCtrackPt_RecoIsDangling","MCtrack's p_{T} (Reco is Dangling);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMtrackPtEta_RecoIsDangling = new TH2F("recoGMtrackPtEta_RecoIsDangling","reconstructed GMtrack's p_{T}-#eta (Reco is Dangling);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	TH2F *perfectGMtrackPtEta_RecoIsDangling = new TH2F("perfectGMtrackPtEta_RecoIsDangling","perfect GMtrack's p_{T}-#eta (Reco is Dangling);p_{T}^{perfect}[GeV/c];#eta^{perfect}",200,0,10,200,-4.0,-2.0);
	TH2F *MCtrackPtEta_RecoIsDangling = new TH2F("MCtrackPtEta_RecoIsDangling","MCtrack's p_{T}-#eta (Reco is Dangling);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);

  recoGMTrackAllPtEta->SetOption("COLZ");
  perfectGMTrackAllPtEta->SetOption("COLZ");
  MCtrackAllPtEta->SetOption("COLZ");
	recoGMtrackPtEta_RecoIsPairable->SetOption("COLZ");
  perfectGMtrackPtEta_RecoIsPairable->SetOption("COLZ");
  MCtrackPtEta_RecoIsPairable->SetOption("COLZ");
  recoGMtrackPtEta_RecoIsClose->SetOption("COLZ");
  perfectGMtrackPtEta_RecoIsClose->SetOption("COLZ");
  MCtrackPtEta_RecoIsClose->SetOption("COLZ");

  perfectGMtrackPtEta_PerfectIsClose->SetOption("COLZ");
  MCtrackPtEta_PerfectIsClose->SetOption("COLZ");
  perfectGMtrackPtEta_PerfectIsNotClose->SetOption("COLZ");
  MCtrackPtEta_PerfectIsNotClose->SetOption("COLZ");
	recoGMtrackPtEta_RecoIsCorrect->SetOption("COLZ");
  perfectGMtrackPtEta_RecoIsCorrect->SetOption("COLZ");
  MCtrackPtEta_RecoIsCorrect->SetOption("COLZ");
  recoGMtrackPtEta_RecoIsFake->SetOption("COLZ");
  perfectGMtrackPtEta_RecoIsFake->SetOption("COLZ");
  MCtrackPtEta_RecoIsFake->SetOption("COLZ");
  MCtrackPtEta_RecoIsFakeInPairable->SetOption("COLZ");

  recoGMtrackPtEta_RecoIsDangling->SetOption("COLZ");
  perfectGMtrackPtEta_RecoIsDangling->SetOption("COLZ");
  MCtrackPtEta_RecoIsDangling->SetOption("COLZ");

  // Counters
  Int_t nChargeMatch = 0;
  Int_t nChargeMiss = 0;
  Int_t nChargeMatch0_1 = 0;
  Int_t nChargeMiss0_1 = 0;
  Int_t nChargeMatch1_4 = 0;
  Int_t nChargeMiss1_4 = 0;
  Int_t nChargeMatch4plus = 0;
  Int_t nChargeMiss4plus = 0;
  Int_t nCorrectMatchGMTracks = 0;
  Int_t nFakeGMTracks = 0;
  Int_t nNoMatchGMTracks = 0;

  // Files & Trees
  // MC
  TFile* o2sim_KineFileIn = new TFile(o2sim_KineFile.c_str());
  TTree* o2SimKineTree = (TTree*)o2sim_KineFileIn->Get("o2sim");

  vector<MCTrackT<float>>* mcTr = nullptr;
  o2SimKineTree->SetBranchAddress("MCTrack", &mcTr);
  o2::dataformats::MCEventHeader* eventHeader = nullptr;
  o2SimKineTree->SetBranchAddress("MCEventHeader.", &eventHeader);

  Int_t numberOfEvents = o2SimKineTree->GetEntries();

  // Global Muon Tracks
  TFile* trkFileIn = new TFile(trkFile.c_str());
  TTree* gmTrackTree = (TTree*)trkFileIn->Get("o2sim");
  std::vector<GlobalMuonTrack> trackGMVec, *trackGMVecP = &trackGMVec;
  gmTrackTree->SetBranchAddress("GlobalMuonTrack", &trackGMVecP);

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcLabels = nullptr;
  gmTrackTree->SetBranchAddress("GlobalMuonTrackMCTruth", &mcLabels);

  MatchingHelper *matching_helperPtr, matching_helper;
  gDirectory->GetObject("Matching Helper", matching_helperPtr);
  matching_helper = *matching_helperPtr;

  std::string annotation = matching_helper.Annotation();
  std::cout << "matching_helper.Generator = " << matching_helper.Generator
            << std::endl;
  std::cout << "matching_helper.GeneratorConfig = "
            << matching_helper.GeneratorConfig << std::endl;
  std::cout << "matching_helper.MatchingFunction = "
            << matching_helper.MatchingFunction << std::endl;
  std::cout << "matching_helper.MatchingCutFunc = "
            << matching_helper.MatchingCutFunc << std::endl;
  std::cout << "matching_helper.MatchingCutConfig = "
            << matching_helper.MatchingCutConfig << std::endl;
  std::cout << "Annotation = " << annotation << std::endl;

  // Perfect Global Muon Tracks
  TFile* perfecttrkFileIn = new TFile(perfecttrkFile.c_str());
  TTree* perfectGMtrackTree = (TTree*)perfecttrkFileIn->Get("o2sim");
  std::vector<GlobalMuonTrack> trackPerfectGMVec, *trackPerfectGMVecP = &trackPerfectGMVec;//trackPerfectGMVecPはtracPerfectGMVecのアドレスの別名
  perfectGMtrackTree->SetBranchAddress("GlobalMuonTrack", &trackPerfectGMVecP);//trackPerfectGMVecPのアドレス(=trackPerfectGMVecのアドレスのアドレス)にGlobalMuonTrackというブランチの値をセット
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcLabels_perfect = nullptr;
  perfectGMtrackTree->SetBranchAddress("GlobalMuonTrackMCTruth", &mcLabels_perfect);

  // MFT Tracks
  TFile* mfttrkFileIn = new TFile("mfttracks.root");
  TTree* mftTrackTree = (TTree*)mfttrkFileIn->Get("o2sim");
  // std::vector<o2::mft::TrackMFT> trackMFTVec, *trackMFTVecP = &trackMFTVec;
  // mftTrackTree->SetBranchAddress("MFTTrack", &trackMFTVecP);

  std::vector<o2::MCCompLabel>* mftMcLabels = nullptr;
  mftTrackTree->SetBranchAddress("MFTTrackMCTruth", &mftMcLabels);
  mftTrackTree->GetEntry(0);

  gmTrackTree->GetEntry(0);
  o2SimKineTree->GetEntry(0);

  auto field_z = getZField(0, 0, -61.4); // Get field at Center of MFT

  std::string outfilename = "GlobalMuonChecks.root";
  TFile outFile(outfilename.c_str(), "RECREATE");

  // Reconstructed Global Muon Tracks
  std::cout << "Loop over events and reconstructed Global Muon Tracks!"
            << std::endl;
  // GMTracks - Identify reconstructed tracks
  auto nCloseMatches = 0;
  for (int iEvent = 0; iEvent < numberOfEvents; iEvent++) {
    auto iTrack = 0;
    if (DEBUG_VERBOSE) {
      std::cout << "Event = " << iEvent << " with " << trackGMVec.size()
                << " MCH tracks " << std::endl;
    }
    o2SimKineTree->GetEntry(iEvent);
    gmTrackTree->GetEntry(iEvent);
    perfectGMtrackTree->GetEntry(iEvent);

    if (0)
      for (auto& gmTrack : trackGMVec) {
        const auto& label = mcLabels->getLabels(iTrack);
        std::cout << "iTrack = " << iTrack;
        label[0].print();
        iTrack++;
      }

    for (auto& gmTrack : trackGMVec) {
      const auto& label = mcLabels->getLabels(iTrack);
      auto iTrack_perfect = 0;
      for (auto& perfectGMtrack : trackPerfectGMVec){
				const auto& label_perfect = mcLabels_perfect->getLabels(iTrack_perfect);
      	auto bestMFTTrackMatchID = gmTrack.getBestMFTTrackMatchID();
      	// std::cout << "iTrack = " << iTrack;
      	// label[0].print();
      	if (iEvent == label[0].getEventID() && iEvent == label_perfect[0].getEventID() && label[0].getTrackID() == label_perfect[0].getTrackID()) {
	//std::cout<<"==========================================================================================================================================================="<<endl;
	//std::cout<<"gmTrack track ID = "<<label[0].getTrackID()<<" ; gmTrack Event ID = "<<label[0].getEventID()<<" ; perfect Track ID = "<<label_perfect[0].getTrackID()<<" ; perfect Event ID = "<<label_perfect[0].getEventID()<<endl;

				auto thisTrkID = label_perfect[0].getTrackID();
        MCTrackT<float>* thisTrack = &(*mcTr).at(thisTrkID);
	/*
        if (DEBUG_VERBOSE) {
          // std::cout << "  Global Track ID = " <<  iTrack << " ; MFTMatchID =
          // " << bestMFTTrackMatchID << " SourceID = " <<
          // label[0].getSourceID()
          // << " ; EventID = " << label[0].getEventID() << ":  trackID = " <<
          // label[0].getTrackID() << " ; isFake = " << label[0].isFake() << "
          // Label: ";
          std::cout << "  Global Track ID = " << iTrack
                    << " ; MFTMatchID = " << bestMFTTrackMatchID << " Label: ";
          label[0].print();

          // std::cout << "        bestMFTTrackMatchID = " <<
          // bestMFTTrackMatchID << " / labelMFTBestMatch = ";
          // labelMFTBestMatch[0].print();
        }
	*/
	/*
	std::cout<<"Output Matching fundamental Info"<<endl;
	std::cout << "Global Track ID = " <<  iTrack << " ; isClose = " << gmTrack.closeMatch() << " ; MFTMatchID = " << bestMFTTrackMatchID << " ; SourceID = " <<label[0].getSourceID()<< " ; EventID = " << label[0].getEventID() << " ; trackID = " <<label[0].getTrackID() << " ; isFake = " << label[0].isFake() << " Label: "<<endl;
	std::cout<<"Output MCtrack's fundamental Info"<<endl;
	std::cout<<"vx_MC = "<< thisTrack->GetStartVertexCoordinatesX() <<" ; vy_MC = "<< thisTrack->GetStartVertexCoordinatesY() <<" ; vz_MC = "<< thisTrack->GetStartVertexCoordinatesZ() <<" ; Pt_MC = "<< thisTrack->GetPt() <<" ; P_MC = "<< thisTrack->GetP() <<" ; phi_MC = "<< TMath::ATan2(thisTrack->Py(), thisTrack->Px()) <<" ; GetPhi() = "<< thisTrack->GetPhi() << " ; eta_MC = "<< atanh(thisTrack->GetStartVertexMomentumZ() / thisTrack->GetP()) <<" ; GetEta() = "<< thisTrack->GetEta() << " ; tanl_MC = "<< thisTrack->Pz() / thisTrack->GetPt() <<" ; pdgcode_MC = "<< thisTrack->GetPdgCode() <<" ; isPrimary = "<< thisTrack->isPrimary() <<endl;
	*/

	//// All GMtracks ////////////////////////////////////////////////////////////////////////////////
	recoGMTrackAllPt->Fill(gmTrack.getPt());
	perfectGMTrackAllPt->Fill(perfectGMtrack.getPt());
	MCtrackAllPt->Fill(thisTrack->GetPt());
	recoGMTrackAllPtEta->Fill(gmTrack.getPt(),gmTrack.getEta());
	perfectGMTrackAllPtEta->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
	MCtrackAllPtEta->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));

	//// Reconstructed GMtrack is Pairable ////////////////////////////////////////////////////////////////////////////////
	if(gmTrack.pairable()){
		recoGMtrackPt_RecoIsPairable->Fill(gmTrack.getPt());
		recoGMtrackPtEta_RecoIsPairable->Fill(gmTrack.getPt(),gmTrack.getEta());
		perfectGMtrackPt_RecoIsPairable->Fill(perfectGMtrack.getPt());
		perfectGMtrackPtEta_RecoIsPairable->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
		MCtrackPt_RecoIsPairable->Fill(thisTrack->GetPt());
		MCtrackPtEta_RecoIsPairable->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
	}
	if(!(gmTrack.pairable())){
		recoGMtrackPt_RecoIsNotPairable->Fill(gmTrack.getPt(),gmTrack.getEta());
	}

	//// Reconstrucrted GMtrack is Close ////////////////////////////////////////////////////////////////////////////////
  if (gmTrack.closeMatch()){
		recoGMtrackPt_RecoIsClose->Fill(gmTrack.getPt());
		recoGMtrackPtEta_RecoIsClose->Fill(gmTrack.getPt(),gmTrack.getEta());
		perfectGMtrackPt_RecoIsClose->Fill(perfectGMtrack.getPt());
		perfectGMtrackPtEta_RecoIsClose->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
		MCtrackPt_RecoIsClose->Fill(thisTrack->GetPt());
		MCtrackPtEta_RecoIsClose->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
    nCloseMatches++;
	}
	if (!gmTrack.closeMatch()){
	  recoGMtrackPt_RecoIsNotClose->Fill(gmTrack.getPt(),gmTrack.getEta());
	}

	//// Perfect GMtrack is Close ////////////////////////////////////////////////////////////////////////////////
	if (perfectGMtrack.closeMatch()){
	  perfectGMtrackPt_PerfectIsClose->Fill(perfectGMtrack.getPt());
	  perfectGMtrackPtEta_PerfectIsClose->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
	  MCtrackPt_PerfectIsClose->Fill(thisTrack->GetPt());
	  MCtrackPtEta_PerfectIsClose->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
		/*
			if (label[0].isCorrect()==1 && !(gmTrack.getPt()-perfectGMtrack.getPt()==0)){
	      closePt_perfect_replace->Fill(gmTrack.getPt());
	    }
	    else {
	      closePt_perfect_replace->Fill(perfectGMtrack.getPt());
	    }
		*/
	}
	if (!perfectGMtrack.closeMatch()){
	  perfectGMtrackPtEta_PerfectIsNotClose->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
	  MCtrackPtEta_PerfectIsNotClose->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
	}

	//// Fill Histograms exist from first ////////////////////////////////////////////////////////////////////////////////
        pairedMCHTracksEff->Fill(bestMFTTrackMatchID > -1, gmTrack.getPt());
        globalMuonCombinedEff->Fill(label[0].isCorrect(), gmTrack.getPt());
        closeMatchEff->Fill(gmTrack.closeMatch(), gmTrack.getPt());

        if (bestMFTTrackMatchID >= 0) {
          globalMuonCorrectMatchRatio->Fill(label[0].isCorrect(),
                                            gmTrack.getPt());
        }

 //// Reconstructed GMtrack is Correct ////////////////////////////////////////////////////////////////////////////////
        if (label[0].isCorrect()) { // Correct match track: add to histograms
	  /*
	  if (gmTrack.getPt()-perfectGMtrack.getPt()){
	    std::cout<<" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  "<<endl;
	    std::cout<<"Output Reconstructed GMtrack's fundamental Info"<<endl;

	    std::cout<<"delta = "<<fixed<<setprecision(100)<<gmTrack.getPt()-perfectGMtrack.getPt()<<endl;

	    std::cout<<"Global Track pT = "<<fixed<<setprecision(100)<<gmTrack.getPt()<<endl;
	    std::cout<<"Perfec Track pT = "<<fixed<<setprecision(100)<<perfectGMtrack.getPt()<<endl;

	  }
	  */
	  			recoGMtrackPt_RecoIsCorrect->Fill(gmTrack.getPt());
	  			perfectGMtrackPt_RecoIsCorrect->Fill(perfectGMtrack.getPt());
	  			MCtrackPt_RecoIsCorrect->Fill(thisTrack->GetPt());
	  			recoGMtrackPtEta_RecoIsCorrect->Fill(gmTrack.getPt(),gmTrack.getEta());
	  			perfectGMtrackPtEta_RecoIsCorrect->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
          MCtrackPtEta_RecoIsCorrect->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
          nCorrectMatchGMTracks++;
          // pairedMCHTracksEff->Fill(1,gmTrack.getPt());
          auto vx_MC = thisTrack->GetStartVertexCoordinatesX();
          auto vy_MC = thisTrack->GetStartVertexCoordinatesY();
          auto vz_MC = thisTrack->GetStartVertexCoordinatesZ();
          auto Pt_MC = thisTrack->GetPt();
          auto P_MC = thisTrack->GetP();
          auto phi_MC = TMath::ATan2(thisTrack->Py(), thisTrack->Px());
          auto eta_MC =
            atanh(thisTrack->GetStartVertexMomentumZ() / P_MC); // eta;
          auto tanl_MC = thisTrack->Pz() / thisTrack->GetPt();
          auto pdgcode_MC = thisTrack->GetPdgCode();
          // std::cout << "pdgcode_MC = " <<  pdgcode_MC;
          int Q_MC;
          if (TDatabasePDG::Instance()->GetParticle(pdgcode_MC)) {
            Q_MC =
              TDatabasePDG::Instance()->GetParticle(pdgcode_MC)->Charge() / 3;
            if (DEBUG_VERBOSE)
              std::cout << "      => "
                        << TDatabasePDG::Instance()
                             ->GetParticle(pdgcode_MC)
                             ->GetName()
                        << "\n";
          }

          else {
            Q_MC = 0;
            std::cout << " => pdgcode ERROR " << Q_MC << "\n";
          }

          gmTrack.propagateToZhelix(vz_MC, field_z);
          // gmTrack.propagateToZquadratic(vz_MC,field_z);
          // gmTrack.propagateToZlinear(vz_MC,field_z);
          auto Q_fit = gmTrack.getCharge();
          auto dx = gmTrack.getX() - vx_MC;
          auto dy = gmTrack.getY() - vy_MC;
          auto d_eta = gmTrack.getEta() - eta_MC;
          auto d_tanl = gmTrack.getTanl() - tanl_MC;
          auto Pt_fit = gmTrack.getPt();
          auto d_invQPt = Q_fit / Pt_fit - Q_MC / Pt_MC;
          auto P_fit = gmTrack.getP();
          auto P_res = P_fit / P_MC;
          auto Pt_res = Pt_fit / Pt_MC;
          auto d_Phi = gmTrack.getPhi() - phi_MC;
          auto d_Charge = Q_fit - Q_MC;
          auto xChi2 = dx * dx / gmTrack.getCovariances()(0, 0);
          auto yChi2 = dy * dy / gmTrack.getCovariances()(1, 1);
          auto phiChi2 = d_Phi * d_Phi / gmTrack.getCovariances()(2, 2);
          auto tanlChi2 = d_tanl * d_tanl / gmTrack.getCovariances()(3, 3);
          auto invQPtChi2 =
            d_invQPt * d_invQPt / sqrt(gmTrack.getCovariances()(4, 4));
          auto fitChi2 = xChi2 + yChi2 + phiChi2 + tanlChi2; // + invQPtChi2;
          auto trackChi2 = gmTrack.getTrackChi2();
          TH1Histos[kGMTracksP]->Fill(gmTrack.getP());
          TH1Histos[kGMTrackDeltaTanl]->Fill(d_tanl);
          TH1Histos[kGMTrackDeltaPhi]->Fill(d_Phi);
          TH1Histos[kGMTrackDeltaInvQPt]->Fill(d_invQPt);
          TH1Histos[kGMTrackDeltaPhiDeg]->Fill(TMath::RadToDeg() * d_Phi);
          TH1Histos[kGMTrackDeltaX]->Fill(dx);

          // std::cout << "DeltaX / sigmaX = " <<
          // dx/sqrt(gmTrack.getCovariances()(0,0)) << std::endl;
          TH1Histos[kGMTrackDeltaXErr]->Fill(
            dx / sqrt(gmTrack.getCovariances()(0, 0)));
          // std::cout << "DeltaY / sigmaY = " <<
          // dy/sqrt(gmTrack.getCovariances()(1,1)) << std::endl;
          TH1Histos[kGMTrackDeltaYErr]->Fill(
            dy / sqrt(gmTrack.getCovariances()(1, 1)));
          // std::cout << "DeltaPhi / sigmaPhi = " <<
          // d_Phi/sqrt(gmTrack.getCovariances()(2,2)) << std::endl;
          TH1Histos[kGMTrackDeltaPhiErr]->Fill(
            d_Phi / sqrt(gmTrack.getCovariances()(2, 2)));
          // std::cout << "DeltaTanl / sigmaTanl = " <<
          // d_tanl/sqrt(gmTrack.getCovariances()(3,3)) << std::endl;
          TH1Histos[kGMTrackDeltaTanLErr]->Fill(
            d_tanl / sqrt(gmTrack.getCovariances()(3, 3)));
          // std::cout << "DeltaPt / sigmaPt = " <<
          // d_Pt/sqrt(gmTrack.getCovariances()(4,4)) << std::endl;
          TH1Histos[kGMTrackDeltainvQPtErr]->Fill(
            d_invQPt / sqrt(gmTrack.getCovariances()(4, 4)));

          //
          TH1Histos[kMCHResTrackDeltaXErr]->Fill(gmTrack.getResiduals2Cov()(0));
          TH1Histos[kMCHResTrackDeltaYErr]->Fill(gmTrack.getResiduals2Cov()(1));
          TH1Histos[kMCHResTrackDeltaPhiErr]->Fill(
            gmTrack.getResiduals2Cov()(2));
          TH1Histos[kMCHResTrackDeltaTanLErr]->Fill(
            gmTrack.getResiduals2Cov()(3));
          TH1Histos[kMCHResTrackDeltainvQPtErr]->Fill(
            gmTrack.getResiduals2Cov()(4));

          TH1Histos[kGMTrackXChi2]->Fill(xChi2);
          TH1Histos[kGMTrackYChi2]->Fill(yChi2);
          TH1Histos[kGMTrackPhiChi2]->Fill(phiChi2);
          TH1Histos[kGMTrackTanlChi2]->Fill(tanlChi2);
          TH1Histos[kGMTrackinvQPtChi2]->Fill(invQPtChi2);
          TH1Histos[kFitChi2]->Fill(fitChi2);
          TH2Histos[kGMTrackChi2vsFitChi2]->Fill(fitChi2, trackChi2);

          DeltaX_Profile->Fill(Pt_MC, dx * 1e4);
          TH1Histos[kGMTrackDeltaY]->Fill(dy);
          TH1Histos[kGMTrackR]->Fill(sqrt(dx * dx + dy * dy));
          TH1Histos[kGMTrackQ]->Fill(d_Charge);
          TH1Histos[kGMTrackChi2]->Fill(trackChi2);
          TH2Histos[kGMTrackDeltaXYVertex]->Fill(10. * dx, 10. * dy);
          TH2Histos[kGMTrackQPRec_MC]->Fill(P_MC * Q_MC, P_fit * Q_fit);
          TH2Histos[kGMTrackPtResolution]->Fill(Pt_MC, Pt_fit / Pt_MC);
          PtRes_Profile->Fill(Pt_MC, Pt_fit / Pt_MC);
          TH2Histos[kGMTrackInvPtResolution]->Fill(
            Pt_MC, (1.0 / Pt_fit - 1.0 / Pt_MC) * Pt_MC);

          // MC histos
          TH1Histos[kMCTrackspT]->Fill(Pt_MC);
          TH1Histos[kMCTracksp]->Fill(P_MC);
          TH1Histos[kMCTrackEta]->Fill(eta_MC);
          TH2Histos[kMCTracksEtaZ]->Fill(vz_MC, eta_MC);

          // Differential histos
          if (Pt_MC <= 1.0) {
            TH2Histos[kGMTrackDeltaXYVertex0_1]->Fill(10. * dx, 10. * dy);
            TH1Histos[kGMTrackDeltaTanl0_1]->Fill(d_tanl);
            TH1Histos[kGMTrackDeltaPhi0_1]->Fill(d_Phi);
            TH1Histos[kGMTrackDeltaPhiDeg0_1]->Fill(TMath::RadToDeg() * d_Phi);
            TH1Histos[kGMTrackDeltaX0_1]->Fill(dx);
            TH1Histos[kGMTrackQ0_1]->Fill(d_Charge);
            d_Charge ? nChargeMiss0_1++ : nChargeMatch0_1++;
          }
          if (Pt_MC > 1.0 and Pt_MC <= 4) {
            TH2Histos[kGMTrackDeltaXYVertex1_4]->Fill(10. * dx, 10. * dy);
            TH1Histos[kGMTrackDeltaTanl1_4]->Fill(d_tanl);
            TH1Histos[kGMTrackDeltaPhi1_4]->Fill(d_Phi);
            TH1Histos[kGMTrackDeltaPhiDeg1_4]->Fill(TMath::RadToDeg() * d_Phi);
            TH1Histos[kGMTrackDeltaX1_4]->Fill(dx);
            TH1Histos[kGMTrackQ1_4]->Fill(d_Charge);
            d_Charge ? nChargeMiss1_4++ : nChargeMatch1_4++;
          }
          if (Pt_MC > 4.0) {
            TH2Histos[kGMTrackDeltaXYVertex4plus]->Fill(10. * dx, 10. * dy);
            TH1Histos[kGMTrackDeltaTanl4plus]->Fill(d_tanl);
            TH1Histos[kGMTrackDeltaPhi4plus]->Fill(d_Phi);
            TH1Histos[kGMTrackDeltaPhiDeg4plus]->Fill(TMath::RadToDeg() *
                                                      d_Phi);
            TH1Histos[kGMTrackDeltaX4plus]->Fill(dx);
            TH1Histos[kGMTrackQ4plus]->Fill(d_Charge);
            d_Charge ? nChargeMiss4plus++ : nChargeMatch4plus++;
          }

          d_Charge ? nChargeMiss++ : nChargeMatch++;
          qMatchEff->Fill(!d_Charge, Pt_MC);
        } else {
          if (bestMFTTrackMatchID >= 0) {
	    recoGMtrackPt_RecoIsFake->Fill(gmTrack.getPt());
	    perfectGMtrackPt_RecoIsFake->Fill(perfectGMtrack.getPt());
	    MCtrackPt_RecoIsFake->Fill(thisTrack->GetPt());
	    recoGMtrackPtEta_RecoIsFake->Fill(gmTrack.getPt(),gmTrack.getEta());
	    perfectGMtrackPtEta_RecoIsFake->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
      MCtrackPtEta_RecoIsFake->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
			if(gmTrack.pairable()){
				MCtrackPt_RecoIsFakeInPairable->Fill(thisTrack->GetPt());
				MCtrackPtEta_RecoIsFakeInPairable->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
			}
      nFakeGMTracks++;
          } else{
	    recoGMtrackPt_RecoIsDangling->Fill(gmTrack.getPt());
	    perfectGMtrackPt_RecoIsDangling->Fill(perfectGMtrack.getPt());
	    MCtrackPt_RecoIsFake->Fill(thisTrack->GetPt());
	    recoGMtrackPtEta_RecoIsDangling->Fill(gmTrack.getPt(),gmTrack.getEta());
	    perfectGMtrackPtEta_RecoIsDangling->Fill(perfectGMtrack.getPt(),perfectGMtrack.getEta());
      MCtrackPtEta_RecoIsDangling->Fill(thisTrack->GetPt(),atanh(thisTrack->GetStartVertexMomentumZ()/thisTrack->GetP()));
	    nNoMatchGMTracks++;
	  }
        }
      }
      iTrack_perfect++;
      }//Loop on Perfect
      iTrack++;
      //}//Loop on Perfect
    } // Loop on GMTracks
  }   // Loop over events

  Int_t nRecoGMTracks = nCorrectMatchGMTracks + nFakeGMTracks;
  Int_t nMCHTracks = nRecoGMTracks + nNoMatchGMTracks;

  // Customize histograms
  TH1Histos[kGMTrackQ]->SetTitle(
    Form("nChargeMatch = %d (%.2f%%)", nChargeMatch,
         100. * nChargeMatch / (nChargeMiss + nChargeMatch)));
  TH1Histos[kGMTrackQ0_1]->SetTitle(
    Form("nChargeMatch = %d (%.2f%%)", nChargeMatch0_1,
         100. * nChargeMatch0_1 / (nChargeMiss0_1 + nChargeMatch0_1)));
  TH1Histos[kGMTrackQ1_4]->SetTitle(
    Form("nChargeMatch = %d (%.2f%%)", nChargeMatch1_4,
         100. * nChargeMatch1_4 / (nChargeMiss1_4 + nChargeMatch1_4)));
  TH1Histos[kGMTrackQ4plus]->SetTitle(
    Form("nChargeMatch = %d (%.2f%%)", nChargeMatch4plus,
         100. * nChargeMatch4plus / (nChargeMiss4plus + nChargeMatch4plus)));

  qMatchEff->SetTitle(Form("Charge match = %.2f%%",
                           100. * nChargeMatch / (nChargeMiss + nChargeMatch)));
  pairedMCHTracksEff->SetTitle(
    Form("Paired_MCH_tracks_=_%.2f%%", 100. * nRecoGMTracks / (nMCHTracks)));
  globalMuonCorrectMatchRatio->SetTitle(
    Form("Correct_Match_Ratio = %.2f%%",
         100. * nCorrectMatchGMTracks / (nRecoGMTracks)));
  closeMatchEff->SetTitle(
    Form("Close_Match_=_%.2f%%", 100. * nCloseMatches / (nMCHTracks)));

  // Remove stat boxes
  TH2Histos[kGMTrackQPRec_MC]->SetStats(0);
  TH2Histos[kGMTrackPtResolution]->SetStats(0);
  TH2Histos[kGMTrackInvPtResolution]->SetStats(0);
  TH2Histos[kMCTracksEtaZ]->SetStats(0);
  PtRes_Profile->SetStats(0);
  DeltaX_Profile->SetStats(0);
  TH1Histos[kGMTrackQ]->SetStats(0);

  // Fit Slices: Pt resolution
  FitSlicesy(*TH2Histos[kGMTrackInvPtResolution], *TH2Histos[kGMTrackQPRec_MC]);
  FitSlicesy(*TH2Histos[kGMTrackPtResolution], *TH2Histos[kGMTrackQPRec_MC]);

  // sigmaX resultion Profile
  TH1D* DeltaX_Error;
  DeltaX_Error = DeltaX_Profile->ProjectionX("DeltaX_Error", "C=E");
  DeltaX_Error->GetYaxis()->SetTitleOffset(1.25);
  DeltaX_Error->SetMaximum(500);
  // DeltaX_Error->GetYaxis()->SetLimits(0,500.0);

  // Summary Canvases
  // Matching summary
  auto matching_summary = summary_report_3x2(
    *pairedMCHTracksEff, *globalMuonCorrectMatchRatio, *closeMatchEff,
    *TH2Histos[kGMTrackDeltaXYVertex], *DeltaX_Error, *PtRes_Profile,
    "Matching Summary", annotation, 0, 0, 0, 0, 0, 0, "-", "-", "-",
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex]->GetEntries()),
    "-", "-");

  // Parameters resolution
  auto param_resolution = summary_report_3x2(
    *TH2Histos[kGMTrackDeltaXYVertex], *TH2Histos[kGMTrackPtResolution],
    *PtRes_Profile, *DeltaX_Error, *TH2Histos[kGMTrackQPRec_MC], *qMatchEff,
    "Param Summary", annotation, 0, 0, 0, 0, 0, 0,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex]->GetEntries()),
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackPtResolution]->Integral() /
                     TH2Histos[kGMTrackPtResolution]->GetEntries()),
    "-", "-",
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackQPRec_MC]->Integral() /
                     TH2Histos[kGMTrackQPRec_MC]->GetEntries()),
    "-");

  // Covariances summary
  auto covariances_summary = summary_report_3x2(
    *TH1Histos[kGMTrackDeltaXErr], *TH1Histos[kGMTrackDeltaPhiErr],
    *TH1Histos[kGMTrackDeltainvQPtErr], *TH1Histos[kGMTrackDeltaYErr],
    *TH1Histos[kGMTrackDeltaTanLErr], *TH2Histos[kGMTrackQPRec_MC],
    "Covariances Summary", annotation, 1, 1, 1, 1, 1, 0,
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaXErr]->Integral() /
                     TH1Histos[kGMTrackDeltaXErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiErr]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltainvQPtErr]->Integral() /
                     TH1Histos[kGMTrackDeltainvQPtErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaYErr]->Integral() /
                     TH1Histos[kGMTrackDeltaYErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanLErr]->Integral() /
                     TH1Histos[kGMTrackDeltaTanLErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackQPRec_MC]->Integral() /
                     TH2Histos[kGMTrackQPRec_MC]->GetEntries()));

  // MCH Residuals Covariances summary
  auto MCHcovariances_summary = summary_report_3x2(
    *TH1Histos[kMCHResTrackDeltaXErr], *TH1Histos[kMCHResTrackDeltaPhiErr],
    *TH1Histos[kMCHResTrackDeltainvQPtErr], *TH1Histos[kMCHResTrackDeltaYErr],
    *TH1Histos[kMCHResTrackDeltaTanLErr], *TH2Histos[kGMTrackQPRec_MC],
    "MCH residuals Covariances Summary", annotation, 1, 1, 1, 1, 1, 0,
    Form("%.2f%%", 100.0 * TH1Histos[kMCHResTrackDeltaXErr]->Integral() /
                     TH1Histos[kMCHResTrackDeltaXErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kMCHResTrackDeltaPhiErr]->Integral() /
                     TH1Histos[kMCHResTrackDeltaPhiErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kMCHResTrackDeltainvQPtErr]->Integral() /
                     TH1Histos[kMCHResTrackDeltainvQPtErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kMCHResTrackDeltaYErr]->Integral() /
                     TH1Histos[kMCHResTrackDeltaYErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kMCHResTrackDeltaTanLErr]->Integral() /
                     TH1Histos[kMCHResTrackDeltaTanLErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackQPRec_MC]->Integral() /
                     TH2Histos[kGMTrackQPRec_MC]->GetEntries()));

  // Covariances summary 3x3
  auto par_cov_summary3x3 = summary_report_3x3(
    *TH2Histos[kGMTrackDeltaXYVertex], *TH1Histos[kGMTrackDeltaXErr],
    *TH1Histos[kGMTrackDeltaYErr], *DeltaX_Error,
    *TH2Histos[kGMTrackQPRec_MC], *TH1Histos[kGMTrackDeltaPhiErr], *qMatchEff,
    *TH1Histos[kGMTrackDeltainvQPtErr], *TH1Histos[kGMTrackDeltaTanLErr],
    "par_cov_summary3x3", annotation, 0, 1, 1, 0, 0, 1, 0, 1, 1,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaXErr]->Integral() /
                     TH1Histos[kGMTrackDeltaXErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaYErr]->Integral() /
                     TH1Histos[kGMTrackDeltaYErr]->GetEntries()),
    "-",
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackQPRec_MC]->Integral() /
                     TH2Histos[kGMTrackQPRec_MC]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiErr]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiErr]->GetEntries()),
    "-",
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltainvQPtErr]->Integral() /
                     TH1Histos[kGMTrackDeltainvQPtErr]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanLErr]->Integral() /
                     TH1Histos[kGMTrackDeltaTanLErr]->GetEntries()));

  auto param_summary_diff_pt = summary_report_3x3(
    *TH1Histos[kGMTrackDeltaX0_1], *TH1Histos[kGMTrackDeltaTanl0_1],
    *TH1Histos[kGMTrackDeltaPhiDeg0_1], *TH1Histos[kGMTrackDeltaX1_4],
    *TH1Histos[kGMTrackDeltaTanl1_4], *TH1Histos[kGMTrackDeltaPhiDeg1_4],
    *TH1Histos[kGMTrackDeltaX4plus], *TH1Histos[kGMTrackDeltaTanl4plus],
    *TH1Histos[kGMTrackDeltaPhiDeg4plus], "ParamSummaryVsPt", annotation, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX0_1]->Integral() /
                     TH1Histos[kGMTrackDeltaX0_1]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanl0_1]->Integral() /
                     TH1Histos[kGMTrackDeltaTanl0_1]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg0_1]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg0_1]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX1_4]->Integral() /
                     TH1Histos[kGMTrackDeltaX1_4]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanl1_4]->Integral() /
                     TH1Histos[kGMTrackDeltaTanl1_4]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg1_4]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg1_4]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX4plus]->Integral() /
                     TH1Histos[kGMTrackDeltaX4plus]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanl4plus]->Integral() /
                     TH1Histos[kGMTrackDeltaTanl4plus]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg4plus]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg4plus]->GetEntries()));

  auto pt_resolution = summary_report(
    *TH2Histos[kGMTrackPtResolution], *TH2Histos[kGMTrackQPRec_MC],
    *PtRes_Profile, *qMatchEff, "Pt Summary", annotation, 0, 0, 0, 0,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackPtResolution]->Integral() /
                     TH2Histos[kGMTrackPtResolution]->GetEntries()),
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackQPRec_MC]->Integral() /
                     TH2Histos[kGMTrackQPRec_MC]->GetEntries()));

  auto invpt_resolution = summary_report(
    *TH2Histos[kGMTrackInvPtResolution], *TH2Histos[kGMTrackQPRec_MC],
    *(TH1F*)gDirectory->Get(
      (std::string(TH2Histos[kGMTrackInvPtResolution]->GetName()) +
       std::string("_1"))
        .c_str()),
    *(TH1F*)gDirectory->Get(
      (std::string(TH2Histos[kGMTrackInvPtResolution]->GetName()) +
       std::string("_2"))
        .c_str()),
    "InvPt Summary", annotation, 0, 0, 0, 0,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackInvPtResolution]->Integral() /
                     TH2Histos[kGMTrackInvPtResolution]->GetEntries()),
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackQPRec_MC]->Integral() /
                     TH2Histos[kGMTrackQPRec_MC]->GetEntries()));

  auto vertexing_resolution = summary_report(
    *TH2Histos[kGMTrackDeltaXYVertex], *TH1Histos[kGMTrackDeltaX],
    *DeltaX_Error, *TH1Histos[kGMTrackDeltaPhiDeg], "Vertexing Summary",
    annotation, 0, 1, 0, 1,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX]->Integral() /
                     TH1Histos[kGMTrackDeltaX]->GetEntries()),
    Form("-"),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg]->GetEntries()));

  auto vertexing_resolution0_1 = summary_report(
    *TH2Histos[kGMTrackDeltaXYVertex0_1], *TH1Histos[kGMTrackDeltaX0_1],
    *TH1Histos[kGMTrackDeltaTanl0_1], *TH1Histos[kGMTrackDeltaPhiDeg0_1],
    "Vertexing Summary pt < 1", annotation, 0, 1, 1, 1,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex0_1]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex0_1]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX0_1]->Integral() /
                     TH1Histos[kGMTrackDeltaX0_1]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanl0_1]->Integral() /
                     TH1Histos[kGMTrackDeltaTanl0_1]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg0_1]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg0_1]->GetEntries()));

  auto vertexing_resolution1_4 = summary_report(
    *TH2Histos[kGMTrackDeltaXYVertex1_4], *TH1Histos[kGMTrackDeltaX1_4],
    *TH1Histos[kGMTrackDeltaTanl1_4], *TH1Histos[kGMTrackDeltaPhiDeg1_4],
    "Vertexing Summary 1 < p_t < 4", annotation, 0, 1, 1, 1,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex1_4]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex1_4]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX1_4]->Integral() /
                     TH1Histos[kGMTrackDeltaX1_4]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanl1_4]->Integral() /
                     TH1Histos[kGMTrackDeltaTanl1_4]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg1_4]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg1_4]->GetEntries()));

  auto vertexing_resolution4plus = summary_report(
    *TH2Histos[kGMTrackDeltaXYVertex4plus], *TH1Histos[kGMTrackDeltaX4plus],
    *TH1Histos[kGMTrackDeltaTanl4plus], *TH1Histos[kGMTrackDeltaPhiDeg4plus],
    "Vertexing Summary p_t > 4", annotation, 0, 1, 1, 1,
    Form("%.2f%%", 100.0 * TH2Histos[kGMTrackDeltaXYVertex4plus]->Integral() /
                     TH2Histos[kGMTrackDeltaXYVertex4plus]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaX4plus]->Integral() /
                     TH1Histos[kGMTrackDeltaX4plus]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaTanl4plus]->Integral() /
                     TH1Histos[kGMTrackDeltaTanl4plus]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackDeltaPhiDeg4plus]->Integral() /
                     TH1Histos[kGMTrackDeltaPhiDeg4plus]->GetEntries()));

  auto chi2_summary = summary_report(
    *TH1Histos[kGMTrackChi2], *TH1Histos[kGMTrackXChi2],
    *TH1Histos[kGMTrackTanlChi2], *TH1Histos[kGMTrackPhiChi2], "Chi2 Summary",
    annotation, 1, 1, 1, 1,
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackChi2]->Integral() /
                     TH1Histos[kGMTrackChi2]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackXChi2]->Integral() /
                     TH1Histos[kGMTrackXChi2]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackTanlChi2]->Integral() /
                     TH1Histos[kGMTrackTanlChi2]->GetEntries()),
    Form("%.2f%%", 100.0 * TH1Histos[kGMTrackPhiChi2]->Integral() /
                     TH1Histos[kGMTrackPhiChi2]->GetEntries()));

  // Write histograms to file and export images

  outFile.mkdir("MoreHistos");
  outFile.cd("MoreHistos");

  for (auto& h : TH2Histos) {
    h->Write();
    if (EXPORT_HISTOS_IMAGES)
      exportHisto(*h);
  }

  for (auto& h : TH1Histos) {
    h->Write();
    if (EXPORT_HISTOS_IMAGES)
      exportHisto(*h);
  }
  //allPt->Add(CorrectGMtrackspT,FakeGMtrackspT);
  //allPt->Add(GMtrackspT,DanglingtrackspT);
  PtRes_Profile->Write();
  DeltaX_Profile->Write();
  DeltaX_Error->Write();
  qMatchEff->Write();
  pairedMCHTracksEff->Write();
  globalMuonCorrectMatchRatio->Write();
  closeMatchEff->Write();
  globalMuonCombinedEff->Write();

  recoGMTrackAllPt->Write();
  perfectGMTrackAllPt->Write();
	MCtrackAllPt->Write();
  recoGMTrackAllPtEta->Write();
  perfectGMTrackAllPtEta->Write();
  MCtrackAllPtEta->Write();

  recoGMtrackPt_RecoIsPairable->Write();
  recoGMtrackPtEta_RecoIsPairable->Write();
  perfectGMtrackPt_RecoIsPairable->Write();
  perfectGMtrackPtEta_RecoIsPairable->Write();
	MCtrackPt_RecoIsPairable->Write();
  MCtrackPtEta_RecoIsPairable->Write();

  recoGMtrackPt_RecoIsNotPairable->Write();

  recoGMtrackPt_RecoIsClose->Write();
  recoGMtrackPtEta_RecoIsClose->Write();
  perfectGMtrackPt_RecoIsClose->Write();
  perfectGMtrackPtEta_RecoIsClose->Write();
  MCtrackPt_RecoIsClose->Write();
  MCtrackPtEta_RecoIsClose->Write();

  recoGMtrackPt_RecoIsNotClose->Write();

	perfectGMtrackPt_PerfectIsClose->Write();
  perfectGMtrackPtEta_PerfectIsClose->Write();
  MCtrackPt_PerfectIsClose->Write();
  MCtrackPtEta_PerfectIsClose->Write();

  perfectGMtrackPtEta_PerfectIsNotClose->Write();
  MCtrackPtEta_PerfectIsNotClose->Write();

  recoGMtrackPt_RecoIsCorrect->Write();
  perfectGMtrackPt_RecoIsCorrect->Write();
	MCtrackPt_RecoIsCorrect->Write();
  recoGMtrackPtEta_RecoIsCorrect->Write();
  perfectGMtrackPtEta_RecoIsCorrect->Write();
  MCtrackPtEta_RecoIsCorrect->Write();

	recoGMtrackPt_RecoIsFake->Write();
	perfectGMtrackPt_RecoIsFake->Write();
  MCtrackPt_RecoIsFake->Write();
  recoGMtrackPtEta_RecoIsFake->Write();
  perfectGMtrackPtEta_RecoIsFake->Write();
  MCtrackPtEta_RecoIsFake->Write();

	recoGMtrackPt_RecoIsDangling->Write();
  perfectGMtrackPt_RecoIsDangling->Write();
  MCtrackPt_RecoIsDangling->Write();
  recoGMtrackPtEta_RecoIsDangling->Write();
  perfectGMtrackPtEta_RecoIsDangling->Write();
  MCtrackPtEta_RecoIsDangling->Write();

	TH1F *MCtrackPt_RecoIsCorrectOrFakeInPairable = new TH1F("MCtrackPt_RecoIsCorrectOrFakeInPairable","MCtrack's p_{T} (Reco is Correct or Fake, Pairable);p_{T}^{MC}[GeV/c];Entry",1000,0,10);
	TH2F *MCtrackPtEta_RecoIsCorrectOrFakeInPairable = new TH2F("MCtrackPtEta_RecoIsCorrectOrFakeInPairable","MCtrack's p_{T}-#eta (Reco is Correct or Fake, Pairable);p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4.0,-2.0);
	MCtrackPt_RecoIsCorrectOrFakeInPairable->Add(MCtrackPt_RecoIsCorrect,MCtrackPt_RecoIsFakeInPairable);
	MCtrackPtEta_RecoIsCorrectOrFakeInPairable->Add(MCtrackPtEta_RecoIsCorrect,MCtrackPtEta_RecoIsFakeInPairable);
	MCtrackPtEta_RecoIsCorrectOrFakeInPairable->SetOption("COLZ");
	MCtrackPt_RecoIsCorrectOrFakeInPairable->Write();
	MCtrackPtEta_RecoIsCorrectOrFakeInPairable->Write();

	TH1F *MCtrackPt_RecoIsCorrectOrFake = new TH1F("MCtrackPt_RecoIsCorrectOrFake","Reconstructed GMtrack's p_{T} (Reco is Correct or Fake);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH2F *MCtrackPtEta_RecoIsCorrectOrFake = new TH2F("MCtrackPtEta_RecoIsCorrectOrFake","Reconstructed GMtrack's p_{T}-#eta (Reco is Correct or Fake);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	MCtrackPt_RecoIsCorrectOrFake->Add(MCtrackPt_RecoIsCorrect,MCtrackPt_RecoIsFake);
	MCtrackPtEta_RecoIsCorrectOrFake->Add(MCtrackPtEta_RecoIsCorrect,MCtrackPtEta_RecoIsFake);
	MCtrackPtEta_RecoIsCorrectOrFake->SetOption("COLZ");
	MCtrackPt_RecoIsCorrectOrFake->Write();
	MCtrackPtEta_RecoIsCorrectOrFake->Write();

	TH1F *recoGMtrackPt_RecoIsCorrectOrFake = new TH1F("recoGMtrackPt_RecoIsCorrectOrFake","Reconstructed GMtrack's p_{T} (Reco is Correct or Fake);p_{T}^{reco}[GeV/c];Entry",1000,0,10);
	TH2F *recoGMtrackPtEta_RecoIsCorrectOrFake = new TH2F("recoGMtrackPtEta_RecoIsCorrectOrFake","Reconstructed GMtrack's p_{T}-#eta (Reco is Correct or Fake);p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4.0,-2.0);
	recoGMtrackPt_RecoIsCorrectOrFake->Add(recoGMtrackPt_RecoIsCorrect,recoGMtrackPt_RecoIsFake);
	recoGMtrackPtEta_RecoIsCorrectOrFake->Add(recoGMtrackPtEta_RecoIsCorrect,recoGMtrackPtEta_RecoIsFake);
	recoGMtrackPtEta_RecoIsCorrectOrFake->SetOption("COLZ");
	recoGMtrackPt_RecoIsCorrectOrFake->Write();
	recoGMtrackPtEta_RecoIsCorrectOrFake->Write();

  std::cout<<"1.Rebining N_pairable^MC"<<endl;
  MCtrackPt_RecoIsPairable->Rebin(40);
  std::cout<<"2.Rebining N_All^reco"<<endl;
  recoGMTrackAllPt->Rebin(40);
  std::cout<<"3.Rebining N_True^MC"<<endl;
  MCtrackPt_RecoIsCorrect->Rebin(40);
  std::cout<<"4.Rebining N_Fake^MC"<<endl;
  MCtrackPt_RecoIsFake->Rebin(40);
	std::cout<<"4.5.Rebining N_Fake^MC"<<endl;
  MCtrackPt_RecoIsFakeInPairable->Rebin(40);
  std::cout<<"5.Rebining N_True^reco"<<endl;
  recoGMtrackPt_RecoIsCorrect->Rebin(40);
  std::cout<<"6.Rebining N_Fake^reco"<<endl;
  recoGMtrackPt_RecoIsFake->Rebin(40);
  std::cout<<"7.Rebining N_Close^MC"<<endl;
  MCtrackPt_RecoIsClose->Rebin(40);
	std::cout<<"8.Rebining N_RecoAndPairable^MC"<<endl;
	MCtrackPt_RecoIsCorrectOrFakeInPairable->Rebin(40);
	std::cout<<"9.Rebining N_Reconstructed^reco"<<endl;
	recoGMtrackPt_RecoIsCorrectOrFake->Rebin(40);

  /*
`// Use TH1F for Efficiency
  TH1F *PairingEfficiency = new TH1F("PairingEfficiency","Pairing Efficiency;p_{T}[GeV/c];#epsilon^{GM}_{pairing}",25,0,10);
  TH1F *TruePairingEfficiency = new TH1F("TruePairingEfficiency","True Pairing Efficiency;p_{T}[GeV/c];#epsilon^{GM}_{true}",25,0,10);
  TH1F *FakePairingEfficiency = new TH1F("FakePairingEfficiency","Fake Pairing Efficiency;p_{T}[GeV/c];#epsilon^{GM}_{fake}",25,0,10);
  TH1F *AlternativePairingEfficiency = new TH1F("AlternativePairingEfficiency","Alternative Pairing Efficiency;p_{T}[GeV/c];#epsilon^{MCH/MFT}_{pairing}",25,0,10);
  TH1F *GlobalPairingPurity = new TH1F("GlobalPairingPurity","Global Pairing Purity;p_{T}[GeV/c];P^{GM}_{pairing}",25,0,10);
  TH1F *ClosingMatchingEfficiency = new TH1F("ClosingMatchingEfficiency","Closing Matching Efficiency;p_{T}[GeV/c];#epsilon_{close}",25,0,10);
  PairingEfficiency->Divide(recoGMPt,pairablePt_perfect);
  TruePairingEfficiency->Divide(correctPt,pairablePt_perfect);
  FakePairingEfficiency->Divide(fakePt,pairablePt_perfect);
  AlternativePairingEfficiency->Divide(recoGMPt,allPt);
  GlobalPairingPurity->Divide(correctPt,recoGMPt);
  ClosingMatchingEfficiency->Divide(pairablePt,pairablePt_perfect);
  for (int i=0; i<PairingEfficiency->GetNbinsX()+1; i++){
    PairingEfficiency->SetBinError(i,recoGMPt->GetBinError(i)/pairablePt_perfect->GetBinContent(i));
    TruePairingEfficiency->SetBinError(i,correctPt->GetBinError(i)/pairablePt_perfect->GetBinContent(i));
    FakePairingEfficiency->SetBinError(i,fakePt->GetBinError(i)/pairablePt_perfect->GetBinContent(i));
    AlternativePairingEfficiency->SetBinError(i,recoGMPt->GetBinError(i)/allPt->GetBinContent(i));
    GlobalPairingPurity->SetBinError(i,correctPt->GetBinError(i)/recoGMPt->GetBinContent(i));
    ClosingMatchingEfficiency->SetBinError(i,pairablePt->GetBinError(i)/pairablePt_perfect->GetBinContent(i));
  }
  */

	TH1F *FakePairingEfficiency = new TH1F("FakePairingEfficiency","Fake Pairing Efficiency;p_{T}^{MC}[GeV/c];#epsilon^{GM}_{fake}",25,0,10);
	FakePairingEfficiency->Divide(MCtrackPt_RecoIsFake, MCtrackPt_RecoIsPairable);
	TH1F *PairingEfficiency = new TH1F("PairingEfficiency","Pairing Efficiency;p_{T}^{MC}[GeV/c];#epsilon^{GM}_{pairing}",25,0,10);
	PairingEfficiency->Divide(MCtrackPt_RecoIsCorrectOrFake,MCtrackPt_RecoIsPairable);
	for (int i=0; i<PairingEfficiency->GetNbinsX()+1; i++){
		PairingEfficiency->SetBinError(i,MCtrackPt_RecoIsCorrectOrFake->GetBinError(i)/MCtrackPt_RecoIsPairable->GetBinContent(i));
		FakePairingEfficiency->SetBinError(i,MCtrackPt_RecoIsFake->GetBinError(i)/MCtrackPt_RecoIsPairable->GetBinContent(i));
	}

  //Use TEfficiency for Efficiency
  std::cout<<"making PairingEfficiency In Pairable"<<endl;
  TEfficiency *PairingEfficiencyInPairable = new TEfficiency(*MCtrackPt_RecoIsCorrectOrFakeInPairable,*MCtrackPt_RecoIsPairable);
  std::cout<<"making TruePairingEfficiency"<<endl;
  TEfficiency *TruePairingEfficiency = new TEfficiency(*MCtrackPt_RecoIsCorrect,*MCtrackPt_RecoIsPairable);
  std::cout<<"making FakePairingEfficiency In Pairable"<<endl;
  TEfficiency *FakePairingEfficiencyInPairable = new TEfficiency(*MCtrackPt_RecoIsFakeInPairable,*MCtrackPt_RecoIsPairable);
  std::cout<<"making AlternativePairingEfficiency"<<endl;
  TEfficiency *AlternativePairingEfficiency = new TEfficiency(*MCtrackPt_RecoIsCorrectOrFake,*recoGMTrackAllPt);
  std::cout<<"making GlobalPairingPurity"<<endl;
  TEfficiency *GlobalPairingPurity = new TEfficiency(*recoGMtrackPt_RecoIsCorrect,*recoGMtrackPt_RecoIsCorrectOrFake);
  std::cout<<"making ClosingMatchingEfficiency"<<endl;
  TEfficiency *ClosingMatchingEfficiency = new TEfficiency(*MCtrackPt_RecoIsClose,*MCtrackPt_RecoIsPairable);

  PairingEfficiencyInPairable->SetTitle("Pairing Efficiency (InPairable);p_{T}^{MC}[GeV/c];#epsilon^{GM}_{pairing}");
  TruePairingEfficiency->SetTitle("True Pairing Efficiency;p_{T}^{MC}[GeV/c];#epsilon^{GM}_{true}");
  FakePairingEfficiencyInPairable->SetTitle("Fake Pairing Efficiency (InPairable);p_{T}^{MC}[GeV/c];#epsilon^{GM}_{fake}");
  AlternativePairingEfficiency->SetTitle("Alternative Pairing Efficiency;p_{T}^{reco}[GeV/c];#epsilon^{MCH/MFT}_{pairing}");
  GlobalPairingPurity->SetTitle("Global Pairing Purity;p_{T}^{reco}[GeV/c];P^{GM}_{pairing}");
  ClosingMatchingEfficiency->SetTitle("Closing Matching Efficiency;p_{T}^{MC}[GeV/c];#epsilon_{close}");

  //Write Efficiency
  PairingEfficiency->Write();
	PairingEfficiencyInPairable->Write();
  TruePairingEfficiency->Write();
  FakePairingEfficiency->Write();
	FakePairingEfficiencyInPairable->Write();
  AlternativePairingEfficiency->Write();
  GlobalPairingPurity->Write();
  ClosingMatchingEfficiency->Write();


  //Use TH2F for pT-Eta Efficiency
  TH2F *PairingEfficiencyPtEta = new TH2F("PairingEfficiencyPtEta","Pairing Efficiency;p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4,-2);
  TH2F *TruePairingEfficiencyPtEta = new TH2F("TruePairingEfficiencyPtEta","True Pairing Efficiency;p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4,-2);
  TH2F *FakePairingEfficiencyPtEta = new TH2F("FakePairingEfficiencyPtEta","Fake Pairing Efficiency;p_{T}^{MC}[GeV/c];#eta^{MC}",200,0,10,200,-4,-2);
  TH2F *AlternativePairingEfficiencyPtEta = new TH2F("AlternativePairingEfficiencyPtEta","Alternative Pairing Efficiency;p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4,-2);
  TH2F *GlobalPairingPurityPtEta = new TH2F("GlobalPairingPurityPtEta","Global Pairing Purity;p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4,-2);
  TH2F *ClosingMatchingEfficiencyPtEta = new TH2F("ClosingMatchingEfficiencyPtEta","Closing Matching Efficiency;p_{T}^{reco}[GeV/c];#eta^{reco}",200,0,10,200,-4,-2);
  PairingEfficiencyPtEta->Divide(MCtrackPtEta_RecoIsCorrectOrFakeInPairable,MCtrackPtEta_RecoIsPairable);
  TruePairingEfficiencyPtEta->Divide(MCtrackPtEta_RecoIsCorrect,MCtrackPtEta_RecoIsPairable);
  FakePairingEfficiencyPtEta->Divide(MCtrackPtEta_RecoIsFake,MCtrackPtEta_RecoIsPairable);
  AlternativePairingEfficiencyPtEta->Divide(MCtrackPtEta_RecoIsCorrectOrFakeInPairable,recoGMtrackPtEta_RecoIsCorrectOrFake);
  GlobalPairingPurityPtEta->Divide(recoGMtrackPtEta_RecoIsCorrect,recoGMtrackPtEta_RecoIsCorrectOrFake);
  ClosingMatchingEfficiencyPtEta->Divide(MCtrackPtEta_RecoIsClose,MCtrackPtEta_RecoIsPairable);
  PairingEfficiencyPtEta->SetOption("COLZ");
  TruePairingEfficiencyPtEta->SetOption("COLZ");
  FakePairingEfficiencyPtEta->SetOption("COLZ");
  AlternativePairingEfficiencyPtEta->SetOption("COLZ");
  GlobalPairingPurityPtEta->SetOption("COLZ");
  ClosingMatchingEfficiencyPtEta->SetOption("COLZ");

  /*
  //Use TEfficiency for pT-Eta Efficiency
  //std::cout<<"making PairingEfficiencyPtEta"<<endl;
  //TEfficiency *PairingEfficiencyPtEta = new TEfficiency(*recoGMPtEta,*pairablePtEta_perfect);
  std::cout<<"making TruePairingEfficiencyPtEta"<<endl;
  TEfficiency *TruePairingEfficiencyPtEta = new TEfficiency(*correctPtEta,*pairablePtEta_perfect);
  //std::cout<<"making FakePairingEfficiencyPtEta"<<endl;
  //TEfficiency *FakePairingEfficiencyPtEta = new TEfficiency(*fakePtEta,*pairablePtEta_perfect);
  std::cout<<"making AlternativePairingEfficiencyPtEta"<<endl;
  TEfficiency *AlternativePairingEfficiencyPtEta = new TEfficiency(*recoGMPtEta,*allPtEta);
  std::cout<<"making GlobalPairingPurityPtEta"<<endl;
  TEfficiency *GlobalPairingPurityPtEta = new TEfficiency(*correctPtEta,*recoGMPtEta);
  std::cout<<"making ClosingMatchingEfficiencyPtEta"<<endl;
  TEfficiency *ClosingMatchingEfficiencyPtEta = new TEfficiency(*pairablePtEta_perfect_reco,*pairablePtEta_perfect);

  //PairingEfficiencyPtEta->SetTitle("Pairing Efficiency;p_{T}[GeV/c];#eta");
  TruePairingEfficiencyPtEta->SetTitle("True Pairing Efficiency;p_{T}[GeV/c];#eta");
  //FakePairingEfficiencyPtEta->SetTitle("Fake Pairing Efficiency;p_{T}[GeV/c];#eta");
  AlternativePairingEfficiencyPtEta->SetTitle("Alternative Pairing Efficiency;p_{T}[GeV/c];#eta");
  GlobalPairingPurityPtEta->SetTitle("Global Pairing Purity;p_{T}[GeV/c];eta");
  ClosingMatchingEfficiencyPtEta->SetTitle("Closing Matching Efficiency;p_{T}[GeV/c];#eta");
  */

  //Write pT-Eta Efficiency
  PairingEfficiencyPtEta->Write();
  TruePairingEfficiencyPtEta->Write();
  FakePairingEfficiencyPtEta->Write();
  AlternativePairingEfficiencyPtEta->Write();
  GlobalPairingPurityPtEta->Write();
  ClosingMatchingEfficiencyPtEta->Write();

  outFile.cd();
  outFile.WriteObjectAny(&matching_helper, "MatchingHelper", "Matching Helper");

  outFile.Close();

  std::cout << std::endl;
  std::cout << "---------------------------------------------------"
            << std::endl;
  std::cout << "-------------   Matching Summary   ----------------"
            << std::endl;
  std::cout << "---------------------------------------------------"
            << std::endl;
  std::cout << " P_mean = " << TH1Histos[kGMTracksP]->GetMean() << std::endl;
  std::cout << " P_StdDev = " << TH1Histos[kGMTracksP]->GetStdDev()
            << std::endl;
  std::cout << " Tanl_mean = " << TH1Histos[kGMTrackDeltaTanl]->GetMean()
            << std::endl;
  std::cout << " Tanl_StdDev = " << TH1Histos[kGMTrackDeltaTanl]->GetStdDev()
            << std::endl;
  std::cout << " Tanl_StdDev(pt<1) = "
            << TH1Histos[kGMTrackDeltaTanl0_1]->GetStdDev() << std::endl;
  std::cout << " Tanl_StdDev(1<pt<4) = "
            << TH1Histos[kGMTrackDeltaTanl1_4]->GetStdDev() << std::endl;
  std::cout << " Tanl_StdDev(pt>4) = "
            << TH1Histos[kGMTrackDeltaTanl4plus]->GetStdDev() << std::endl;
  std::cout << " Phi_mean = " << TH1Histos[kGMTrackDeltaPhi]->GetMean()
            << std::endl;
  std::cout << " Phi_StdDev = " << TH1Histos[kGMTrackDeltaPhi]->GetStdDev()
            << std::endl;
  std::cout << " Phi_StdDev(pt<1) = "
            << TH1Histos[kGMTrackDeltaPhi0_1]->GetStdDev() << std::endl;
  std::cout << " Phi_StdDev(1<pt<4) = "
            << TH1Histos[kGMTrackDeltaPhi1_4]->GetStdDev() << std::endl;
  std::cout << " Phi_StdDev(pt>4) = "
            << TH1Histos[kGMTrackDeltaPhi4plus]->GetStdDev() << std::endl;
  std::cout << " Phi_meanDeg = " << TH1Histos[kGMTrackDeltaPhiDeg]->GetMean()
            << std::endl;
  std::cout << " Phi_StdDevDeg = "
            << TH1Histos[kGMTrackDeltaPhiDeg]->GetStdDev() << std::endl;
  std::cout << " Phi_StdDevDeg(pt<1) = "
            << TH1Histos[kGMTrackDeltaPhiDeg0_1]->GetStdDev() << std::endl;
  std::cout << " Phi_StdDevDeg(1<pt<4) = "
            << TH1Histos[kGMTrackDeltaPhiDeg1_4]->GetStdDev() << std::endl;
  std::cout << " Phi_StdDevDeg(pt>4) = "
            << TH1Histos[kGMTrackDeltaPhiDeg4plus]->GetStdDev() << std::endl;
  std::cout << " DeltaX_mean = " << TH1Histos[kGMTrackDeltaX]->GetMean()
            << std::endl;
  std::cout << " DeltaX_StdDev = " << TH1Histos[kGMTrackDeltaX]->GetStdDev()
            << std::endl;
  std::cout << " DeltaX_StdDev(pt<1) = "
            << TH1Histos[kGMTrackDeltaX0_1]->GetStdDev() << std::endl;
  std::cout << " DeltaX_StdDev(1<pt<4) = "
            << TH1Histos[kGMTrackDeltaX1_4]->GetStdDev() << std::endl;
  std::cout << " DeltaX_StdDev(pt>4) = "
            << TH1Histos[kGMTrackDeltaX4plus]->GetStdDev() << std::endl;
  std::cout << " DeltaY_mean = " << TH1Histos[kGMTrackDeltaY]->GetMean()
            << std::endl;
  std::cout << " DeltaY_StdDev = " << TH1Histos[kGMTrackDeltaY]->GetStdDev()
            << std::endl;
  std::cout << " R_mean = " << TH1Histos[kGMTrackR]->GetMean() << std::endl;
  std::cout << " R_StdDev = " << TH1Histos[kGMTrackR]->GetStdDev() << std::endl;
  std::cout << " Charge_mean = " << TH1Histos[kGMTrackDeltaY]->GetMean()
            << std::endl;
  std::cout << " nChargeMatch = " << nChargeMatch << " ("
            << 100. * nChargeMatch / (nChargeMiss + nChargeMatch) << "%)"
            << std::endl;
  std::cout << " nTrackMatch = " << nCorrectMatchGMTracks << " ("
            << 100. * nCorrectMatchGMTracks / (nRecoGMTracks) << "%)"
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "------------"
            << std::endl;

  std::cout << std::endl;
  std::cout << "---------------------------------------------------------------"
               "------------"
            << std::endl;
  std::cout << "------------------------   Track matching Summary   "
               "-----------------------"
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "------------"
            << std::endl;
  std::cout << " ==> " << nMCHTracks << " MCH Tracks in " << numberOfEvents
            << " events" << std::endl;
  std::cout << " ==> " << nNoMatchGMTracks
            << " dangling MCH Tracks (no MFT track to match)"
            << " (" << 100. * nNoMatchGMTracks / (nMCHTracks) << "%)"
            << std::endl;
  std::cout << " ==> " << nRecoGMTracks << " reconstructed Global Muon Tracks"
            << " (" << 100. * nRecoGMTracks / (nMCHTracks) << "%)" << std::endl;
  std::cout << " ==> " << nFakeGMTracks << " fake Global Muon Tracks"
            << " (contamination = " << 100. * nFakeGMTracks / (nRecoGMTracks)
            << "%)" << std::endl;
  std::cout << " ==> " << nCloseMatches
            << " close matches - correct MFT track in search window"
            << " (" << 100. * nCloseMatches / (nMCHTracks) << "%)"
            << std::endl;
  std::cout << " ==> " << nCorrectMatchGMTracks
            << " Correct Match Global Muon Tracks"
            << " (Correct_Match_Ratio = "
            << 100. * nCorrectMatchGMTracks / (nRecoGMTracks) << "%)"
            << " (eff. = " << 100. * nCorrectMatchGMTracks / (nMCHTracks)
            << "%)" << std::endl;

  std::cout << "---------------------------------------------------------------"
               "-----------"
            << std::endl;
  std::cout << " Annotation: " << annotation << std::endl;
  std::cout << "---------------------------------------------------------------"
               "-----------"
            << std::endl;
  std::cout << std::endl;

  /*
  std::cout << "matching_helper.nMCHTracks = " << matching_helper.nMCHTracks <<
  std::endl; std::cout << "matching_helper.nNoMatch = " <<
  matching_helper.nNoMatch << std::endl; std::cout <<
  "matching_helper.nGMTracks() = " << matching_helper.nGMTracks() << std::endl;
  std::cout << "matching_helper.nFakes = " << matching_helper.nFakes <<
  std::endl; std::cout << "matching_helper.nCorrectMatches = " <<
  matching_helper.nCorrectMatches << std::endl; std::cout <<
  "matching_helper.nCloseMatches = " << matching_helper.nCloseMatches <<
  std::endl; std::cout << "matching_helper.getCorrectMatchRatio() = " <<
  matching_helper.getCorrectMatchRatio() << std::endl; std::cout <<
  "matching_helper.getPairingEfficiency() = " <<
  matching_helper.getPairingEfficiency() << std::endl;
  */

  return 0;
}
