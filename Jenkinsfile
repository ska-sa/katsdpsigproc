#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.setDependencies(['ska-sa/katsdpdockerbase/master'])
katsdp.standardBuild(opencl: true, cuda: true)
katsdp.mail('sdpdev+katsdpsigproc@sarao.ac.za')
