#!/usr/bin/env Rscript
#RAPL_API_DIR='/sys/class/powercap/intel-rapl'
#/Users/rajrupachattaraj/MyProjectEnergy/R

#Getting socket_id list
check_api=file.exists("/sys/class/powercap/intel-rapl")
socket_id_list <- list()

socket_id = 0
RAPL_API_DIR="/sys/class/powercap/intel-rapl"
name <- paste(RAPL_API_DIR,"/intel-rapl:",socket_id, sep ="")

get_socket_id_list <- function(){
  if(file.exists(name)){

    socket_id_list<- append(socket_id_list,socket_id)

    socket_id = socket_id + 1
    socket_id_list<- append(socket_id_list,socket_id)
  }
  else{
    print("Sorry! Host machine not supported..")
    quit(save="yes")
  }
  return(socket_id_list)
}




#Getting available package domain
socket_id_list<-list()
available_package_domains<-list()
#available_package_domains <- list()
get_available_package_domains <- function(){

  socket_id_list<-get_socket_id_list()

  for (socket_id in socket_id_list) {

    domain_name_file_str = paste(RAPL_API_DIR,"/intel-rapl:",socket_id,"/name", sep="")

    # c1=file.exists("/sys/class/powercap/intel-rapl/intel-rapl:1/name")

    if(file.exists(domain_name_file_str)){

      domain_name_file <- file(domain_name_file_str)
      r1 <- readLines(domain_name_file, n=1)
      package_domain <- paste("package-", socket_id, sep="")
      #print(package_domain)

      if(package_domain==r1){

        available_package_domains<-append(available_package_domains,socket_id)
        #print(available_package_domains)
      }

    }

  }
  return(available_package_domains)

}

#Getting DRAM domains

#available_dram_domains <- list()

get_available_dram_domains <- function(){
  socket_id_list<-get_socket_id_list()
  dram_var1="dram"
  dram_var2="core"
  for (socket_id in socket_id_list) {

    domain_id=0

    domain_name_file_str = paste(RAPL_API_DIR,"/intel-rapl:",socket_id,"/intel-rapl:",socket_id,":",domain_id,"/name",sep ="")
    #print(domain_name_file_str)
    c1=file.exists("/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/name")

    if(file.exists(domain_name_file_str)){

      domain_name_file <- file(domain_name_file_str)
      r1 <- readLines(domain_name_file, n=1)
      #print(r1)


      if(dram_var1==r1 || dram_var2==r1){

        available_dram_domains<-append(available_dram_domains,socket_id)

        #print(available_dram_domains)
      }
      domain_id=domain_id+1
    }
  }
  return(available_dram_domains)
}



#Get Domain file name and energy_uj
value=""
package_domain_value=""
energies_package <- list()
energies_dram <- list()
available_package_domains<-list()
available_dram_domains<-list()
get_energy_dram <- function(){
  dram_var1="dram"
  dram_var2="core"
  available_dram_domains<- get_available_dram_domains()

  for (socket_id in available_dram_domains) {
    domain_id=0

    domain_name_file_str = paste(RAPL_API_DIR,"/intel-rapl:",socket_id,"/intel-rapl:",socket_id,":",domain_id,"/name",sep ="")
    if(file.exists(domain_name_file_str)){

      domain_name_file <- file(domain_name_file_str)
      r1 <- readLines(domain_name_file, n=1)
      #print(r1)
      if(r1==dram_var1 || r1==dram_var2){
        #print("Checking Dram")
        value<- paste(RAPL_API_DIR,"/intel-rapl:",socket_id,"/intel-rapl:",socket_id,":",domain_id,"/energy_uj",sep ="")

        check=file.exists(value)


        energies_dram1<- readLines(value,n=1)

        energies_dram<-append(energies_dram,energies_dram1)
      } else{
        domain_id=domain_id+1
      }
    }
  }

  return(energies_dram)


}


get_energy_package<- function(){
  available_package_domains<- get_available_package_domains()

  #print(available_package_domains)
  for (socket_id in available_package_domains) {
    package_domain_value <- paste(RAPL_API_DIR,'/intel-rapl:',socket_id,'/energy_uj',sep = "")
    #check1=file.exists(package_domain_value)
    #print("This is package domain...")
    #print(check1)

    #file_name_package <- file(package_domain_value)
    #print(package_domain_value)

    energies_package1<- readLines(package_domain_value,n=1)

    #print(energies_package)
    energies_package<-append(energies_package,energies_package1)
    #print("Checking iteration")
  }

  return(energies_package)
}
#library(tinsel)
trace_list_dram<-list()
trace_list_package<-list()
trace_total_list<-list()
get_final_energy_trace<-function(){
  val_dram0=0
  val_dram1=0
  val_package0=0
  val_package1=0
  energies_dram<-get_energy_dram()
  tryCatch(
    expr = {val_dram0=energies_dram[[1]]},
    error = function(e){
      val_dram0=val_dram0
    }
  )

  #print(ncol(energies_dram))
  tryCatch(
    expr = {val_dram1=energies_dram[[2]]},
    error = function(e){
      val_dram1=val_dram1
    }
)

  trace_list_dram<-append(val_dram0,val_dram1)

  energies_package<-get_energy_package()

  tryCatch(
    expr = {val_package0=energies_package[[1]]},
    error = function(e){
      val_package0=val_package0
    }
  )
  tryCatch(
    expr = {val_package1=energies_package[[2]]},
    error = function(e){
      val_package1=val_package1
    }
  )

  trace_list_package<-append(val_package0,val_package1)


  trace_total_list<-list(trace_list_dram = trace_list_dram, trace_list_package = trace_list_package)
  return(trace_total_list)
}




measure_energy <- function(f)
{


  wrapper <- function(...)
  {


    trace_list_start<-get_final_energy_trace()
    start_time=Sys.time()
    print(paste("Time_stamp: ",start_time))



    f(...)

    #print("Finished function call...")
    end_time=Sys.time()
    trace_list_end<-get_final_energy_trace()

    time_elapsed=end_time - start_time
    print(time_elapsed)
    dram0 = lapply(as.numeric(trace_list_end$trace_list_dram[[1]]),"-",as.numeric(trace_list_start$trace_list_dram[[1]]))
    print(paste0("dram:0 ", dram0))

    dram1 = lapply(as.numeric(trace_list_end$trace_list_dram[[2]]),"-",as.numeric(trace_list_start$trace_list_dram[[2]]))
    print(paste0("dram:1 ",dram1))


    trace_list_end<-get_final_energy_trace()
    package0 = lapply(as.numeric(trace_list_end$trace_list_package[[1]]),"-",as.numeric(trace_list_start$trace_list_package[[1]]))
    print(paste0("package:0 ",package0))

    package1 = lapply(as.numeric(trace_list_end$trace_list_package[[2]]),"-",as.numeric(trace_list_start$trace_list_package[[2]]))
    print(paste0("package:1 ",package1))

    tag=paste0(sys.call())

    new_tag <- gsub('.*\\(', '', tag)
    new_tag <- gsub('\\)', '', new_tag)
    print(new_tag)




    time<-as.numeric(time_elapsed)
    timestamp<-as.numeric(start_time)
    df<- data.frame(new_tag=new_tag,
                    timestamp=timestamp,
                    time=time,
                    package0=package0,
                    package1=package1,
                    dram0=dram0,
                    dram1=dram1)
    write.table(df,file='RJoules_output.csv',col.names = c("new_tag","timestamp","time","package0","package1","dram0","dram1"),sep = ",",row.names=FALSE, fileEncoding = "UTF-8", append = TRUE)


  }

  #new.get_energy()

  return(wrapper)

}

