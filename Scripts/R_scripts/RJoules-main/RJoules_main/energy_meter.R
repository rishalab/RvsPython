# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
EnergyMeter() <- class(a)

  init() <- function(self, devices = list(Device), default_tag){
    self$devices = devices
    self$default_tag = default_tag
    self$last_state = None
    self$first_state = None
  }

  stop() <- function(self){
    new_state = self$measure_new_state('__stop__')
    self$append_new_state(new_state)
  }

  generate_trace <- function(self){
    domains = self$get_domain_list()
    generator = TraceGenerator(self$first_state, domains)
    return TraceGenerator$generate()
  }

  measure_new_state() <- function(self, tag){
    timestamp = timeDate::time()
    values = [device$get_energy() for device in self.devices]

    return EnergyState(timestamp, tag if tag is not None else self$default_tag, values)

  }

  append_new_state() <- function(self, new_state){
    self$last_state.add_next_state(new_state)
    self$last_state = new_state
  }

  get_domain_list <- function(self){
    //create a list
    return (Device$get_configured_domains() for device in self.devices)
  }

EnergyState()<-class()
  init()<-function(self, timestamp, tag, values: List[Dict[str, float]]){
    self.timestamp = timestamp
    self.tag = tag
    self.values = values
    self.next_state = None
  }












measure_energy <- function(func = NULL, domains = NULL){


  decorator_measure_energy <- function(func){
    devices <- DeviceFactory$create_devices(domains)
    energy_meter <- EnergyMeter(devices)

    wrapper_measure <- function(...) {
      energy_meter$start(tag = func$test)
      val <- func(...)
      energy_meter$stop()
      handler$process(energy_meter$get_trace())
      return(val)
    }
    return(wrapper_measure)
  }

  if(is.null(func)){
    # to ensure the working system when you call it with parameters or without parameters
    return(decorator_measure_energy)
  } else {
    return(decorator_measure_energy(func))
  }
}

TraceGenerator() <- class()
  init()<- function(self, first_state, domains){
    self$domains = domains
    self$current_state = first_state
  }
  generate()<-function(self){
    generate_next()<-function(current_state, samples){
      if (current_state$next_state==None){
        return samples
      }

      if (current_state$tag == '__stop__'){
        return generate_next(current_state.next_state, samples)
      }
      sample = self$gen_sample(current_state)
      samples$append(sample)
      return generate_next(current_state$next_state, samples)

    }
    samples = generate_next(self$current_state, [])
    return EnergyTrace(samples)

  }


  gen_sample()<-function(self,state){
    return EnergySample(state$timestamp, state$tag, state$compute_duration(), state$compute_energy(self$domains))

  }




