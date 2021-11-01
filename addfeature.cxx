void addfeature(){
  std::cout<<"Loading DeltaZ.root"<<endl;
  TFile fileDeltaZ("DeltaZ.root");
  TTree* zTree = (TTree*)fileDeltaZ.Get("DeltaZtree");
  Int_t trkIDdeltaZ, evtIDdeltaZ;
  Double_t DeltaZ;
  zTree->SetBranchAddress("MFTTrackID", &trkIDdeltaZ);
  zTree->SetBranchAddress("EventID", &evtIDdeltaZ);
  zTree->SetBranchAddress("Delta_Z", &DeltaZ);

  std::cout<<"Loading MLTrainingPre.root"<<endl;
  TFile fileML("MLTrainingPre.root");
  TTree* matchTreePre = (TTree*)fileML.Get("matchTree");
  Int_t trkIDml, evtIDml;
  matchTreePre->SetBranchAddress("MFTtrackID", &trkIDml);
  matchTreePre->SetBranchAddress("EventID", &evtIDml);

  auto NEntries_Z = zTree->GetEntries();
  auto NEntries_ML = matchTreePre->GetEntries();

  Float_t Delta_Z;
  TFile *fout = new TFile("modMLTraining.root", "recreate");
  TTree *matchTree = matchTreePre->CloneTree();
  TTree *addTree = new TTree("addTree","addTree");
  addTree->Branch("Delta_Z", &Delta_Z, "Delta_Z/F");

  Int_t Counter=0;
  for(Int_t iPair=0 ; iPair < NEntries_ML ; iPair++){
    matchTreePre->GetEntry(iPair);
    matchTree->GetEntry(iPair);
    Int_t InnerCounter =0;
    for(Int_t iTrack=0 ; iTrack < NEntries_Z ; iTrack++){
      zTree->GetEntry(iTrack);
      //std::cout<<"iPair = "<<iPair<<" : iTrack = "<<iTrack<<endl;
      if(evtIDdeltaZ == evtIDml && trkIDdeltaZ == trkIDml){
        //std::cout<<"EventID = "<<evtIDml<<" : MFTtrackID = "<<trkIDml<<endl;
	Delta_Z=DeltaZ;
	Counter = Counter +1;
	InnerCounter = InnerCounter +1;
	std::cout<<"iPair = "<<iPair<<" : iTrack = "<<iTrack<<" : Inner Counter = "<<InnerCounter<<endl;
	if(InnerCounter>=2){
	  std::cout<<"EventID = "<<evtIDml<<" : MFTtrackID = "<<trkIDml<<endl;
	}
	addTree->Fill();
	break;
      }//if(same eventID & MFTtrackID)
    }//for(iTrack)
  }//for(iPair)
  std::cout<<"NEntries_Z = "<<NEntries_Z<<" : NEntries_ML = "<<NEntries_ML<<std::endl;
  std::cout<<"Counter = "<<Counter<<endl;
  fout->cd();
  matchTree->Write();
  addTree->Write();
}
