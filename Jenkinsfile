#!groovy

@Library('katsdpjenkins') _
katsdp.setDependencies(['ska-sa/katsdpdockerbase/master'])
katsdp.standardBuild(opencl: true, cuda: true)
katsdp.mail('bmerry@ska.ac.za')
