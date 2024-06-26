
counties<- c(41067, 53015, 20161, 37085, 48157, 
           28049, 19153, 17167, 31153, 6071, 4013,
           34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
           47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
           32003, 4015, 6025)

HI_restriction<- seq(0.5, 0.9, 0.05)

sink("Run_jobs/Eval_jobs")
for(k in counties){
  county<- k
  for(r_model in c( "mixed_constraints"
    # , "alert_constraints","all_constraints", "no_constraints", "hi_constraints"
    )){
    if(!file.exists(paste0("Summer_results/ORL_basic-NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))){
    cat(
      paste0("python full_evaluation_sb3.py policy_type=NA eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n")
      # , paste0("python full_evaluation_sb3.py policy_type=NA eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n")
      )
      cat(
        # paste0("python full_evaluation_sb3.py policy_type=NWS eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"),
        paste0("python full_evaluation_sb3.py policy_type=basic-NWS eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"))

    cat(
      # paste0("python full_evaluation_sb3.py policy_type=NWS eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"),
      paste0("python full_evaluation_sb3.py policy_type=NWS eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"))
    
    cat(
      # paste0("python full_evaluation_sb3.py policy_type=random eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"),
      paste0("python full_evaluation_sb3.py policy_type=random eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"))
    
    cat(
      # paste0("python full_evaluation_sb3.py policy_type=TK eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"),
      paste0("python full_evaluation_sb3.py policy_type=TK eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"))
    }
    
    for(h in HI_restriction){
      if(!file.exists(paste0("Summer_results/ORL_random_eval_samp-R_samp-W_", r_model, "_Rstr-HI-", h, "_fips_", county, ".csv"))){
        cat(paste0("python full_evaluation_sb3.py policy_type=random eval.val_years=true eval.match_similar=true restrict_days=qhi ",
                   "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, "\n"),
            paste0("python full_evaluation_sb3.py policy_type=random eval.val_years=true eval.match_similar=false restrict_days=qhi ",
                   "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, "\n")
        )
      }
    }
    
    for(h in HI_restriction){
      if(!file.exists(paste0("Summer_results/ORL_AA_eval_samp-R_samp-W_", r_model, "_Rstr-HI-", h, "_fips_", county, ".csv"))){
        cat(paste0("python full_evaluation_sb3.py policy_type=AA eval.val_years=true eval.match_similar=true restrict_days=qhi ",
                   "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, "\n"),
            paste0("python full_evaluation_sb3.py policy_type=AA eval.val_years=true eval.match_similar=false restrict_days=qhi ",
                   "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, "\n")
        )
      }
    }
  }
}

# for(k in counties){
#   county<- k
#   for(r_model in c("alert_constraints","all_constraints", "no_constraints", "hi_constraints"
#   )){
#     cat(paste0("python full_evaluation_sb3.py policy_type=NA eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, " restrict_days=none", "\n"))
#   }
# }

sink()

############ Filling in on FASRC:

bench_results<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")

sink("Run_jobs/Eval_jobs")
for(k in counties){
  county<- k
  for(r_model in c( "mixed_constraints")){
    h<- bench_results$aqhi_ot[which(bench_results$County == k)]
    cat(paste0("python full_evaluation_sb3.py policy_type=AA eval.val_years=true eval.match_similar=false restrict_days=qhi ",
               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, "\n"))
  }
}
sink()
