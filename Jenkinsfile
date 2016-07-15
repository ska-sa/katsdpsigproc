#!groovy

stage 'prepare'
node {
    sshagent ['katpull'] {
        checkout scm
    }
}
