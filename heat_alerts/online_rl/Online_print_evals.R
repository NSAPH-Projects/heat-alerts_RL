
counties<- c(41067, 53015, 20161, 37085, 48157, 
           28049, 19153, 17167, 31153, 6071, 4013,
           34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
           47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
           32003, 4015, 6025)

HI_restriction<- seq(0.5, 0.9, 0.05)

sink("Run_jobs/Eval_jobs")
for(k in counties){
  county<- k
  for(r_model in c("all_constraints", "no_constraints", "hi_constraints")){
    cat(
      # paste0("python old_evaluation_SB3.py policy_type=NA eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"),
      paste0("python old_evaluation_SB3.py policy_type=NA eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"))
    
    cat(
      # paste0("python old_evaluation_SB3.py policy_type=NWS eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"),
      paste0("python old_evaluation_SB3.py policy_type=NWS eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"))
    
    cat(
      # paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"),
      paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"))
    
    cat(
      # paste0("python old_evaluation_SB3.py policy_type=TK eval.val_years=false eval.match_similar=true ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"),
      paste0("python old_evaluation_SB3.py policy_type=TK eval.val_years=true eval.match_similar=false ", "county=", county, " r_model=", r_model, " model_name=", r_model, "\n"))
    
    for(h in HI_restriction){
      cat(paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=false eval.match_similar=true restrict_alerts=true ",
                 "county=", county, " HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, " \n"),
          paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=true eval.match_similar=false restrict_alerts=true ",
                 "county=", county, " HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, " \n")
      )
    }
    
    for(h in HI_restriction){
      cat(paste0("python old_evaluation_SB3.py policy_type=AA eval.val_years=false eval.match_similar=true restrict_alerts=true ",
                 "county=", county, " HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, " \n"),
          paste0("python old_evaluation_SB3.py policy_type=AA eval.val_years=true eval.match_similar=false restrict_alerts=true ",
                 "county=", county, " HI_restriction=", h, " r_model=", r_model, " model_name=", r_model, "_Rstr-HI-", h, " \n")
      )
    }
  }
}
sink()


