#!groovy

stage 'prepare'
node {
    sshagent(['katpull']) {
        deleteDir()
        checkout scm
        katsdp.virtualenv('venv', true) {
            katsdp.installPreRequirements()
            katsdp.installRequirements 'requirements.txt'
            katsdp.installRequirements 'test-requirements.txt'
            stash includes: '**', name: 'source'
        }
    }
}

stage 'test'
parallel 'cuda': {
    node('cuda') {
        deleteDir()
        unstash 'source'
        katsdp.virtualenv('venv') {
            withEnv(['CUDA_DEVICE=0']) { katsdp.nosetests('katsdpsigproc') }
        }
    }
}, 'opencl': {
    node('opencl') {
        deleteDir()
        unstash 'source'
        katsdp.virtualenv('venv') {
            withEnv(['PYOPENCL_CTX=0:0']) { katsdp.nosetests('katsdpsigproc') }
        }
    }
}

stage 'doc'
node {
    deleteDir()
    unstash 'source'
    katsdp.virtualenv('venv') {
        katsdp.installRequirements 'doc-requirements.txt'
        sh 'pip install --no-index ".[doc]"'
        katsdp.makeDocs()
    }
}
