#!groovy

stage 'prepare'
node {
    sshagent(['katpull']) {
        checkout scm
        virtualenv('venv', true) {
            installRequirements 'requirements.txt'
            installRequirements 'test-requirements.txt'
            stash includes: '*', name: 'source'
        }
    }
}

stage 'test'
parallel 'cuda': {
    node 'cuda' {
        unstash 'source'
        virtualenv('venv') {
            withEnv(['CUDA_DEVICE=0']) { runTest() }
        }
    }
}, 'opencl': {
    node 'opencl' {
        unstash 'source'
        virtualenv('venv') {
            withEnv(['PYOPENCL_CTX=0:0']) { runTest() }
        }
    }
}

stage 'doc'
node {
    unstash 'source'
    installRequirements 'doc-requirements.txt'
    sh 'pip install --no-index ".[doc]"'
    sh 'rm -rf doc/_build'
    sh 'make -C doc html'
    publishHTML reportName: 'API docs', reportDir: 'doc/_build/html'
}


def installRequirements(String filename) {
    sh "install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt -r $filename"
}

def virtualenv(String path, boolean create=false, Closure closure) {
    def p = pwd()
    if (create) {
        sh "virtualenv $path"
        closure = { sh 'pip install -r ~/docker-base/pre-requirements.txt'; closure() }
    }
    withEnv(["PATH+VE=$p/$path/bin", "VIRTUAL_ENV=$p/$path"], closure)
}

def runTest() {
    sh 'nosetests --with-xunit --with-coverage --cover-package=katsdpsigproc --cover-xml'
    step([$class: 'JUnitResultArchiver', testResults: 'nosetests.xml'])
}
