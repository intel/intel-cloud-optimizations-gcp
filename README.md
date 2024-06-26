PROJECT NOT UNDER ACTIVE MANAGEMENT

This project will no longer be maintained by Intel.

Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  

Intel no longer accepts patches to this project.

If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  

Contact: webadmin@linux.intel.com

<p align="center">
  <img src="./distributed-training/nlp/images/logo-classicblue-800px.png?raw=true" alt="Intel Logo" width="250"/>
</p>

# Intel® Cloud Optimization Modules for GCP

The Intel Cloud Optimization Modules (ICOMs) for GCP are open-source codebases with codified Intel AI software optimizations and instructions built specifically for GCP. The modules are designed to enable AI developers to maximize the performance and productivity of industry-leading Python machine learning and deep learning libraries on Intel hardware. Each module or reference architecture includes a complete instruction set and all source code published on GitHub. You can check out the full suite of Intel Cloud Optimization Modules [here](https://www.intel.com/content/www/us/en/developer/topic-technology/cloud-optimization.html).

Here are the currently released modules for GCP:

- **[Intel Cloud Optimization Modules for GCP: nanoGPT Distributed Training](distributed-training/nlp)**: Example of distributed fine-tuning training of an LLM across multiple CPUs on GCP.
- **[Intel Cloud Optimization Modules for GCP: XGBoost Kubeflow Pipeline](kubeflow)**: Example of a Kubeflow XGBoost pipeline to predict loan default.

Below, you can learn the basics of setting up a GCP project before jumping into the modules if you haven't done so already. 

## Setting up a GCP Project

Before spinning up any resources on GCP, you need to create a GCP Project. The easiest way to do this is by visiting the [Google Cloud Console](https://console.cloud.google.com/cloud-resource-manager) and clicking "+ Create Project". You need to make sure that you have the "Owner" role for the project. We can ensure that is the case by visiting the [IAM & Admin panel here](https://console.cloud.google.com/iam-admin/iam). You will also need billing enabled for the GCP Project, as we will incur some costs for spinning up resources.

**Google Cloud Shell** <br>
When deploying resources on GCP, it is easy to work in Google's native Cloud Shell than to spin up a Virtual Machine (VM), and it is free! There are many services and permissions that are ready out-of-the-box on the Google Cloud Shell. You can open up a cloud shell on the top right of the browser:

![image](./kubeflow/images/gcp_cloud_shell.png)

Upon first opening up the cloud shell, you should be logged in to your GCP project, but if you aren't, you can set your project with

```bash
PROJECT_ID=<PROJECT_ID>
gcloud config set project $PROJECT_ID
```

where `<PROJECT_ID>` is your GCP project which can be found in the console under [Cloud Overview](https://console.cloud.google.com/home/dashboard). You should now be ready to begin building your module on GCP.
