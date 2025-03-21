24-01-2025

discussion about writing a project proposal:
- need to clarify which grant we apply for 
- figure out what information needs to go into the proposal
- work on AICON in parallel to get initial results that can go into proposal 
- start writing background/motivation section of proposal (literature review/what's new)
- how long are we really willing to work on this project (not for years)
- fellowships are generally for one person only. What about funding options for two people?
- have a sharelatex document to write a grant proposal template
- Dependence of grant proposal on job status (i.e. unemployed, postdoc PhD etc...)
 
FOR NEXT TIME:
- Arjun: Work on filling out grant proposal template with content from presentation as starting point
- Quan: Clean up code. Try to reproduce results shown in presentation in repository. Create issue in github about comin/yac
- Both: Clarify the workflow on github (working on same or different branch)



31-01-2025

Discussion:
- Question about making github repository open source
- Need appropriate name before we make it open source
- Potential names: 1) ICON co pilot, ESM co pilot, deep warming, open warming,
- Words that could go into name: Deep leraning, machine, nodes, GPU, supervisor, assistant, super computer, AI, ESM, climate change, global warming, intelligence, diagnostic, time travel machine, DestinE, Couplearning, **coupler**, dynamical core, intial conditions, external forcing, internal variability, Intelligent VariAIblility, convection, overtunring, mixing, Nodes, cores, CPUS, MPI, HPC, Parameterisation, **Couplearning**, Dual, Interface, Exchange, **Fluxes**, Climate sensitivity, **Feedback**, twin, Adaptor, Plugin, online, training, onion, Layer, eddy, wave, fluid dynaimcs, radiation, Filter, **Distill**, Distillearning, DistilE, Extract, DISTILL, capture,
- Quan spoke to Florian about potential funding. Turns out to be difficult to find funding for two postdocs simulataneously
- Given that, it might be a good idea to speak to Daniel about funding possibilities. Either fellowship plus Daniel as host or full funding by Daniel.

FOR NEXT TIME:
- Quan: Continue working on reproducing results. Add description of workflow to README file. 
- Arjun: Continue writing template grant proposal. Familiarise myself with Quan's code.
- Both: Think of name


21-02-2025

Discussion:
- Decided on MEssE as name (Model Essence Extractor) and used this name for github repository
- Jason Huang's keynote speech about Omniverse and Cosmos with ideas similar to what we're working on (i.e. AI model cosmos learns from the physical model omniverse)

FOR NEXT TIME:
- Arjun: Continue working on grant proposal. Familiarise myself with Quan's code/workflow. Raise issues if anything doesn't work.
- Quan: Clean up machine learning algorithm in python script (comin_plugin.py). Summarise problems so that we can ask for help.
- Both: Think of logo design

28-02-2025

Discussion:
- Quan checked that code to build environment works. Also worked for Arjun.
- Question of how to debug code. Don't want to wait until SLURM job has completed to find out run has crashed.
- Another open open question is how to deal with the ICON grid and how to use CNN on ICON data.
- And how to use data from other domains.  
- Need to find an open source machine learning model that has already been used on ICON data.
- Ask Fabricio for name of person at DKRZ doing machine learning stuff in his project

FOR NEXT TIME:
- Quan: Check python scripts
- Arjun: 

21-03-2025

Discussion:
- Quan says that continuing with project is useful for his career prospects
- Arjun is open to continuing with project after joining the BSH, but needs to see how time constraints develop. Access to Levante should not be a problem for a year or so
- Can potentially ask Helmut for help with interpolation of ICON data onto a regular grid.
- Fixing path errors in build and run scripts
