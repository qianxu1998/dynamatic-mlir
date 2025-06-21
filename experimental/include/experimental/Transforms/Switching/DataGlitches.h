#include "experimental/Transforms/Switching/ProfilingAnalyzer.h"

// This function find all glitching nodes within different MGs, "S", "E", and "T" segments will be ignored

void dataGlitchNodeSearch(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults);
// This function calculates a single glitching value based on the given input operands and node type
int calGlitchValue(int op1, int op2, std::string selNode);

// This function update the glitching value for data base nodes
void dataBaseNodeGlitchUpdate(SwitchingInfo &switchInfo, SCFProfilingResult &profileResults, bool debug);
