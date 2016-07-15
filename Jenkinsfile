#!groovy

stage 'prepare'
node {
    sshagent(['katpull']) {
        checkout scm
        makeVenv 'venv' {
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
        venv 'venv' {
            withEnv(['CUDA_DEVICE=0']) { runTest() }
        }
    }
}, 'opencl': {
    node 'opencl' {
        unstash 'source'
        venv 'venv' {
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


def installRequirements(filename) {
    sh "install-requirements.py -d ~/docker-base/base-requirements.txt -d ~/docker-base/gpu-requirements.txt $filename"
}

def venv(name, closure) {
    withEnv(["PATH+VE=\$PWD/$name/bin", "VIRTUAL_ENV=\$PWD/$name"], closure)
}

def makeVenv(name, closure) {
    sh "virtualenv $name"
    venv(name, { sh 'pip install -r ~/docker-base/pre-requirements.txt'; closure(); })
}

def runTest() {
    sh 'nosetests --with-xunit --with-coverage --cover-package=katsdpsigproc --cover-xml'
    step([$class: 'JUnitResultArchiver', testResults: 'nosetests.xml'])
}
