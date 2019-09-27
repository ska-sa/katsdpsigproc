#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.setDependencies(['ska-sa/katsdpdockerbase/master'])
katsdp.standardBuild(opencl: true, cuda: true, python2: false, python3: true)
katsdp.mail('sdpdev+katsdpsigproc@ska.ac.za')
