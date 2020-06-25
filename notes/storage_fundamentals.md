# Storage Fundamentals

- AWS Storage
    - S3
    - Glacier
    - EC2 Instance Storage
    - EBS
    - Elastic File System
    - RDS
    - Other database services
    - CloudFront
- AWS Data Services
    - AWS Storage Gateway
    - AWS Snowball
- Summary

<br/>

## Amazon Storage

<br/>

### Amazon Simple Storage - S3

- A fully managed object based storage service.
- Highly available
- Highly durable
- Very cost effective
- Widely and easily accessible
- Unlimited storage capacity
- Smallest file size supported = 0 bytes
- Largest file size supported = 5 terabytes

#### Regional Based

When uploading data to S3 you are required to specify the regional location for that data to be placed in. Amazon S3
will then store and duplicate your uploaded data multiple times across multiple available zones within that region to
increase both its durability and availability.

#### Durability and Availability

- Objects stored in S3 have a durability of 99.999999999%
- S3 stores numerous copies of the same data in different AZ's
- The availability of the S3 data object is 99.99%

*Availability* - S3 up-time (i.e. how much of the time you will have access to your data).

*Durability* - The probability of not losing your data

#### S3 Buckets

- Objects are stored in S3 buckets--which are like parent directories
- Bucket names must be globally unique
- Data can be uploaded into the bucket or folders within
- Limitation of 100 buckets per AWS account
- Objects have a unique object key identifying that object
- S3 is not a file system so some features only work at the bucket level--not folder level

#### Storage Classes

- Standard
- Standard-IA (Infrequent Access)
- Intelligent Tiering
- One Zone-IA (Infrequent Access)
- Reduced Redundancy Storage (RSS) - No longer recommended

The main difference is the availability and durability.

[Storage Class Documentation](https://docs.aws.amazon.com/AmazonS3/latest/dev/storage-class-intro.html)

These Classes can be split into categories:
1. Frequently Accessed
   - STANDARD and Reduced Redundancy Storage (RSS)
   - STANDARD is the default storage class
   - RSS is no longer recommended by AWS
2. Infrequently Accessed
   - STANDARD-IA and ONE ZONE-IA
   - Offer the same access speed to that of STANDARD
   - Additional cost to retrieve data
   - ONE ZONE-IA does not replicate data across multiple availability zones
   - ONE ZONE-IA is more cost effective than STANDARD-IA
3. Intelligent Tiering
   - Used for unpredictable access patterns
   - Consists of 2 tiers: Frequently accesses & Infrequently accessed
   - Automatically moves data into the appropriate tier based on access patterns
   - Objects must be larger than 128KB

Choosing a storage class:
1. How often is the data likely to be accessed?
2. How critical is my data?
3. How reproducible is the data?
4. Can it easily be created again if need be?
5. Do I know the access patterns of my data?

#### Security

- Bucket Policies
    - Impose set access controls within specific bucket
    - Constructed as JSON policies
    - Only control access to data in the associated bucket
    - Permissions can be very specific using policy conditions
    - Provide added granularity to buckets access
- Access Control Lists (ACLs)
    - ACLs only control access for users outside of your own AWS account, such as public access
    - ACLs are not as granular as bucket policies
    - The permissions are broad in access, for example 'List objects' and 'Write objects'
- Data Encryption
    - Server-side (SSE) and client-side (CSE) encryption methods
    - SSE-S3 (S3 managed keys)
    - SSE-KMS (KMS managed keys)
    - SSE-C (Customer managed keys)
    - CSE-KMS (KMS managed keys)
    - CSE-C (Customer managed keys)
    - SSL is supported (Secure Socket Layer)

- The main difference between client-side and server-side encryption is the location at which the encryption takes
place. Server-side encryption takes place within a AWS S3 and client side encryption occurs on your client prior to
uploading your objects.

#### Pricing

[Amazon S3 Pricing Page](https://aws.amazon.com/s3/pricing/)

#### Data Management

- Versioning
    - Allows for multiple versions of the same object to exist
    - Useful to recover from accidental deletion, or malicious activity
    - Only the latest version of the object is displayed by default
    - Versioning is not enables by default
    - You can't disable versioning, only suspend it
    - Adds cost for storing multiple versions of the same object
- Lifecycle Rules
    - Provides an automatic method of managing the lifecycle of your data stored
    - Ability to configure specific criteria to automatically move data between storage classes, including Glacier
    or even deleting data completely
    - The time frame is configurable, allowing you to set it for your own requirements

#### Common Uses
- Data Backup
- Static Content & Websites
- Large Datasets
- Integration with other AWS Services
  - Elastic Block Store
    -  EBS volumes perform snapshot backups which are stored on S3
  - AWS CloudTrail
    - CloudTrail log files are automatically sent and stored within preconfigured S3 buckets
  - Amazon CloudFront
    - S3 buckets can be used as a CloudFront origin within a distribution

#### Drawbacks
- Data archiving for long term use
- Dynamic and fast changing data
- File system requirements
- Structured data with queries

<br/>

### Amazon Glacier

- A very low cost, long term, durable storage solution (cold storage) suited for long term backup and archival
  requirements.
- It does not provide instant access to your data.
- Attains 99.999999999% durability by replicating data across multiple availability zones within a single region.
- Storage costs are considerably lower than S3.
- Retrieval of your data can take several hours.

#### Vaults and Archives

- Vaults act as containers for Glacier Archives
- Vaults are regional
- Within each vault you can store data as archives
- Archives can be any object similar to S3
- You can have unlimited archives within you Glacier Vaults

#### Glacier Dashboard

- The Glacier dashboard only allows you to create vaults
- Any operational process to upload or retrieve data *HAS* to be done using code:
  - [The Glacier web service API](https://docs.aws.amazon.com/amazonglacier/latest/dev/amazon-glacier-api.html)
  - [AWS SDKs](https://docs.aws.amazon.com/amazonglacier/latest/dev/using-aws-sdk.html)

#### Moving Data into Glacier

1. Create your vaults
2. Move data into Glacier using the API/SDK or S3 Lifecycle rules

#### Data Retrieval

1. Expedited
    - Used for urgent access to a subset of an Archive
    - Less than 250MB
    - Data available within 1 - 5 mins
    - Cost: $0.03/GB and $0.01/request
2. Standard
    - Used to retrieve and of your Archives, regardless of size
    - Data available within 3 - 5 hours
    - Cost: $0.01.GB and $0.05/1,000 requests
3. Bulk
    - Used to retrieve petabytes of data
    - Data available within 5 - 12 hours
    - The cheapest option for data retrieval
    - Cost: $0.0025/GB and $0.025/1,000 requests.

#### Security

By default Glacier encrypts data using AES-256 encryption algorithm

Addition methods of access control:
- Vault access policies
- Resource based policies
- Applied to a specific vault
- Each vault can only contain 1 vault access policy
- Policy is in JSON format
- Policy contains a *principal* component

Vault Lock Policies:
- Once set they cannot be changed
- Used to help maintain specific governance & compliance

You would use *Vault Access Policies* to govern access control features that may change over time and you would use
*Vault Lock Policies* to help you maintain compliance using access controls that must not be changed.

#### Pricing

[Amazon Glacier Pricing](https://aws.amazon.com/glacier/pricing/)

<br/>

### EC2 Instance Storage

#### Instance Store Volumes

Volumes physically reside on the same host that provides your EC2 instance

Amazon EC2 Instance store volumes act as local drives to an EC2 Instance

- Instance store volumes provide ephemeral (temporary) storage for you EC2 instances.
  - Ephemeral storage means that the block level storage that it provides offers no means of persistency.
    Any data stored on these volumes is considered temporary
- Not recommended for critical or valuable data
- If your instance is either stopped or terminated
  - Any data note stored on that instance store volume associated with this instance will be deleted without any
    means of recovery.
- If your instance was simply rebooted, your data would remain intact.
- Instance store volumes are not available for all instances
- Capacity of instance store volumes increases with the size of the EC2 instance
- Instance store volumes have the same security mechanisms provided by EC2
  - They are not a separate service from EC2

#### Benefits

- No additional cost for storage; it's included in the price of the EC2 instance.
- Offer a very high I/O speed
  - can far exceed those provided by the EBS
- I3 instance Family
  - 3.3 million random read IOPS (Input/output operations per second)
  - 1.4 million write IOPS
- Instance store volumes are ideal as a cache or buffer for rapidly changing data without need for retention
- Often used within a load balancing group, where data is replicated and pooled across the fleet

#### Anti-patterns

Instance store volumes should not be used for:
  - Data that needs to remain persistent
  - Data that needs to be accessed and shared by others

If you need to use block level storage and want to maintain persistency, EBS is recommended.

<br/>

### Amazon Elastic Block Store EBS

- Provides Block level storage to EC2 instances
- Offers persistent and durable data storage
- Greater flexibility than that of instance store volumes
- EBS volumes can be attached to EC2 instances for rapidly changing data
- Used to retain valuable data due to it's persistent qualities
- Operates as a separate service to EC2
- EBS volumes act as network attached storage devices
- Each volume can only be attached to one EC2 instance
- Multiple EBS volumes can be attached to a single EC2 instance
- Data is retained if the EC2 instance is stopped, restarted or terminated

#### EBS Snapshots

EBS offers the ability to provide point in time backup snapshots of the entire volume as and when you need to.
You can manually create a snapshot of your volume at any time, or some code to perform this automatically on
a scheduled basis. Snapshots are stored in S3 and can always be duplicated or recreated.

#### High Availability

Every EBS volume is replicated multiple times within the same availability zone of your region to help prevent
the complete loss of data. This means that your EBS volume itself is only available in a single availability zone.

As a result, should your availability zone fail, you will lose access to your EBS volume. Should this occur, you
can simply recreate the volume from your previous snapshot, which is accessible from all availability zones within
that region, and attach it to another instance in another availability zone.

#### EBS Volume Types

**SSD** (Solid State Drives)
- Suited for work with smaller blocks of data
- Databases using transactional workloads
- Often used for boot volumes on EC2 instances
- Types:
    - General Purpose SSD (GP2)
    - Provisioned IOPS (IO1)

**HDD** (Hard Disk Drives)
- Designed for workloads requiring a high rate of throughput (MB/s)
- Big data processing & logging information
- Larger blocks of data
- Types:
    - Cold HDD (SC1)
    - Throughput Optimized HDD (ST1)

**Volume Type Details**:
- *General Purpose SSD (GP2)*
    - Provides single digit millisecond latency
    - Can burst up to 3,000 IOPS
    - A baseline performance of 3 IOPS up to 10,000 IOPS
    - Throughput of 128 MB/s for volumes up to 170GB
    - Throughput increases to 768 KB/s per GB up to the maximum of 160 MB/s (214+ GB volumes)
- *Provisioned IOPS (IO1)*
    - Delivers predictable performance for I/O intensive workloads
    - Specify IOPS rate during the creation of new RBS volume
    - Volumes attached to EBS optimized instances, will deliver the IOPS defined within 10%, 99.9% of the time
    - Volumes range from 4 - 16 TB
    - The maximum IOPS possible is set to 20,000
- *Cold HDD (SC1)*
    - Offers the lowest cost compared to other volume types
    - Designed for large workloads accesses infrequently
    - High throughput capabilities (MB/s)
    - Can burst to 80 MB/s per TB
    - Delivers 99% of expected throughput
    - Can't use as boot volume for instances
- *Throughput Optimized HDD (ST1)*
    - Designed for frequently accessed data
    - Suited to work with large data sets requiring throughput-intensive workloads
    - Ability to burst up to 250 MB/s
    - Maximum burst 500 MB/s per volume
    - Delivers 99% of the expected throughput

#### Encryption

- EBS offers encryption at rest and in transit
- Encryption is managed by the EBS service itself
- It can be enabled with a checkbox

[Blog: How to encrypt an EBS Volume](https://cloudacademy.com/blog/how-to-encrypt-an-ebs-volume-the-new-amazon-ebs-encryption/)

#### Creating a new EBS Volume

There are 2 ways of creating a new EBS volume from within the Management Console
1. During the creation of a new EC2 instance
2. As a stand alone EBS Volume

#### Changing the size of an EBS Volume

- EBS Volumes are elastically scalable
- Increase the volume size via AWS Management Console
- After increase you must extend the file system on the EC2 instance to utilize the additional storage
- Its possible to resize a volume by creating a new volume from a snapshot.

#### Drawbacks

- Not for temporary storage
- EBS volumes can only be access by one EC2 instance at a time, so are not suited for Multi-instance storage access
- S3 is more suited for very high durability and availability

<br/>

### Amazon Elastic File System

- EFS provides a file level storage service
- EFS is fully managed
- Highly available & durable
- Ability to create shared file systems
- Highly scalable
- Concurrent access by 1,000's of instances
- Limitless capacity (storage elastically grows)
- Regional, spanning multiple availability zones

#### Creating an Elastic File System

- Created in management console

From within the *EFS dashboard*, you select to create a new file system, and then you are required to enter some
configuration information. You must select which *VPC* that this file system will exist in, and once selected, EFS
will automatically create mount targets for you, across the availability zones where you have EC2 instances. These
mount targets allow you to connect to the EFS file system, from your EC2 instances, using a configured mount target IP
address.

When mounting the EFS file system, be aware that it is only compatible with NFS version 4 and 4.1. EFS does not
currently support the Windows operating system. You must insure that your Linux EC2 instance has
the NFS client installed for the mounting process--the NFS client version 4.1 is recommended for this procedure.
For each mount point, you are able to select which subnet the mount point exists in, as well as defining your security
group to control access from what instance level.

The *next step* of creating your file system involves *defining* your *tags*, *performance mode*, and
*encryption settings*. The two different *performance modes* of operations are *general purpose* and *Max I/O*. For
most use cases and requirements, you will *likely be using general purpose*. It has the *lowest latency* out of the two
different modes, and will work with many different application workloads. There is a limitation of this mode, allowing
only up to 7,000 file system operations per second to your EFS file system. If, however, you have huge scale
architectures, or your EFS file system is likely to be *used by many thousands of EC2 instances* concurrently, and
will exceed 7,000 operations per second, then you will need to *consider Max I/O*. This also offers a virtually
unlimited amount of throughput and IOPS, in addition to additional latency to each I/O.

The best way to understand which performance option you need is to run tests alongside your application. If your
application sits comfortably within the upper limit of the 7,000 operations per second, then general purpose will
be best suited, with the added plus point of the lower latency. However, if your testing confirms 7,000 operations
per second may be tight, then select Max I/O. When using the general purpose mode of operations, EFS provides a
CloudWatch metric, PercentIOLimit, which allows you to view your operations per second as a percentage of the top
7,000 limit. This allows you to make the decision to migrate and move to the Max I/O file system, should your
operations be reaching the limit.

You also have the opportunity to implement *encryption of your EFS*, using a simple checkbox, and selecting
your desired CMK. Much like EBS, EFS uses the services offered by the key management service, to provide encryption
of crucial storage. However, at this stage, encryption is only offered at rest, and not in transit.

The *final stage* requires you to *review and create your EFS file system*, based on the configuration that you have
specified.

*Once your file system is create*d, you are *presented* with your *mount target points*, allowing you to *connect* your
*EC2 instances* as required. In addition to having the ability of being able to mount the new EFS file system to your
EC2 instances in your VPC, you can always use these mount points on your on-premise service, as long as you connect
via direct connect, or 3rd party VPN.

#### Moving Data to EFS

If you have existing data in on-premises data center or data already in AWS, such as on EC2 instances and you want to
move that data into EFS, then you can use the *file sync* feature. File sync can be configured from within the EFS
dashboard of the management console, and it allows you to securely manage the transfer of files between an existing
storage location, and your EFS file system via a file sync agent.

#### File Sync

If you require the need to sync source files from your on-premises environment, then you can download
the file sync agent as a VMware ESXi host. If you are syncing source files from within AWS, then it will provide a
community-based AMI, to be used with an EC2 instance. This agent is then configured with your source destination
amount target of your EFS file system details, and logically sits in between them. From within the EFS dashboard,
you can then start the file sync, and monitor it's progress with Amazon CloudWatch.

#### Pricing

[EFS Pricing](https://aws.amazon.com/efs/pricing/)

#### Drawbacks

Not ideal for:
- Data Archiving
- Relational Databases
- Temporary Storage

<br/>

### Amazon RDS

AWS Relational Database Service (RDS) is a managed database service that lets you focus on building your application
storage by taking away the administrative components, such up backups, patches, and replication. It supports a variety
of different relational database builders (such as: Amazon Aurora, PostgreSQL, MySQL, MariaDB, Oracle and SQLServer)
and it offers a reliable infrastructure for running your own database in multiple availability zones.

#### Availability and Durability

- Automatic Backups
- Database Snapshots
- Multi-AZ Deployments
- Automatic Host Replacement

#### Other benefits

- Encryption
    - AWS KMS
    - Amazon cloudHSM
- Resource level permissions

#### Maintenance Window

- Maintenance Window
    - RDS performs maintenance on RDS resources for you
- Multi-AZ Maintenance
    - Amazon RDS will conduct maintenance by following these steps:
        - Perform maintenance on the standby
        - Promote the standby to primary
        - Perform maintenance on the old primary, which becomes the new standby

#### RDS - Best Practices

1. Monitor your memory, CPU and storage usage with *Amazon CloudWatch notifications*.
2. Scale up your DB instance when you are approaching storage capacity limits.
3. Enable Automatic backups and set the backup window to occur during the daily low in WriteIOPS.
4. On a MySQL DB instance, do not create more than 10,000 tables using Provisioned IOPS or 1,000 tables using
standard storage.
5. On a MySQL DB instance, avoid tables in your database growing too large.
    - Partition tables so file size stays under 2 TB limit.

#### RDS - Security

1. Amazon RDS DB Instance access is controlled via *Database Security Groups*.
2. RDS Database Security Groups are not interchangeable with EC2 Security groups.
3. Database Security Groups default to "deny all" access.
4. Database Security groups only allow access to the database server port.
5. Amazon RDS generates an SSL certificate for each DB instance, allowing customers to encrypt their DB Instance
connections for enhanced security.
6. Once an Amazon RDS Db Instance deletion API (DeleteDBInstance) is run, the DB Instance is marked for deletion
and once the instance no longer indicates 'deleting' status, it has been removed.

#### RDS - Security Best Practices

1. Do not use AWS root credentials to manage Amazon RDS resources.
2. Use AWS IAM accounts to control access to Amazon RDS API actions.
3. Assign an individual IAM account to each person who manages RDS resources.
4. Grant each user the minimum set of permissions required to perform his or her duties.
5. Use IAM groups to effectively manage permissions for multiple users.
6. Rotate your IAM credential regularly.

<br/>

### Other Database Services

#### Dynamo DB

- NoSQL key-value data store
- Table scanning is made possible using secondary indexes based on your application search parameters.
- You can update streams which allow you to hook them into item label changes.
- Consider Dynamo DB when your application model is schemaless and nonrelational.
- Can also serve as a persistent session storage mechanism for applications to decouple applications and take away
service state.

#### Elasticache

- Managed in-memory cache service for fast, reliable data access.
- Underlying engines behind ElastiCache are **Memcached** and **Redis**.
    - With the Redis engine you can take advantage of multiple availability zones for high availability and scaling to
      read replicas.
- ElastiCache will automatically detect failed nodes and replace them without manual intervention.
- A typical use case for ElastiCache is low latency access of frequently retrieved data.
    - Think cache database results of data within frequent changes for use in a heavily utilized web application.
- It can serve as temporary storage for compute-intensive workloads or when storing the results from IO intense
  queries or calculations.

#### RedShift

- A fully managed petabyte-scale data warehouse optimized for fast delivery performance with large data sets.
    - So if you're using HSMs, CloudHSM, and AWS management services, you can encrypt your data at rest.
- Fully compliant with a variety of compliance standards, including SOC 1, SOC 2, SOC 3, and PCI DSS Level 1.
- Can query your data using standard SQL commands through ODBC or JDBC connections.
- Integrates with other services, including AWS data pipeline and Kinesis.
- Used to archive large amounts of infrequently used data.
- Good for executing analytical queries on large data sets.
- Also an ideal use case for Elastic Map Reduce jobs that convert unstructured data into structured data.

#### Elastic Map Reduce (EMR)

- A Managed Hadoop framework designed for quickly processing large amounts of data in a really cost-effective way.
- It can dynamically scale across EC2 instances based on how much you want to spend.
- Offers self-healing and fault tolerant processing of your data.
- A common use case for using EMR is to process user behavior data that you might have collected from multiple sources.
- if you're already using Hadoop on premise, then migrating to EMR can offer improved cost and processing with less
administration.
-  Every request made to the EMR API is authenticated, so only authenticated users can create look up or terminate
different job flows.
- When launching customer job flows, Amazon EMR sits up on Amazon EC2 security group, off the Master Node to only
allow external access via SSH.
- To protect customer input and output data sets, Amazon EMR transfers data to and from S3 using SSL.

#### Kinesis

- A fully managed service for processing real-time data streams.
- Can capture terabytes of data, per hour, from over 100,000 different sources.
- Output from Kinesis can be saved to storage such as S3, DynamoDB or RedShift, or ported to services such as EMR or
Lambda among others.
- There's a Kinesis Client Library (KCL) that can be used for integration with your other applications.
    - The KCL helps you consume and process data from an Amazon Kinesis stream.
    - The KCL acts as an intermediary between your record processing logic and Streams.
    - When you start a KCL application, it calls the KCL to instantiate a worker. This call provides the KCL with
    configuration information for the application, such as the stream name and AWS credentials.
    - The KCL is different from the Streams API which you get in the AWS SDKs. The Streams API helps you manage
    Streams (including creating streams, resharding, and putting and getting records), while the KCL provides a layer
    of abstraction specifically for processing data in a consumer role.
- If you need a dashboard that shows updates in real time, Kinesis is a perfect solution since you can combine data
sources from all over including social media.

<br/>

### Amazon Cloud Front

#### Amazon CloudFront

- A Content Delivery Network Service
- Distributes data requested through web traffic closer to the end user via edge locations
- Data is cached temporarily, so durability does not come from here
- Origin data can come from Amazon S3, durability of data is guaranteed here

#### Edge Locations

- AWS edge locations are sites deployed in highly populated areas across the globe.
- Edge locations are not used to deploy infrastructure (EC2/EBS/etc.)
- Edge locations allow the ability to cache data and reduce latency for end user access with services such as Amazon
 CloudFront.

#### Distributions

1. Web Distribution
    - Used to distribute both static and dynamic content
    - Uses both HTTP and HTTPS protocol
    - Allows you to add, remove and update objects
    - Ability to provide live stream functionality on your website
    - Uses an 'origin' to define where the source data is coming from
    - Origins can be a web server, EC2 instance or an S3 bucket
2. RTMP Distribution
    - Real-Time Messaging Protocol
    - Should be used if your focus is to distribute streaming media using Adobe Flash media server's RTMP protocol
    - Allows the end user to start viewing media before the complete file has been downloaded from the edge location
    - The source data can only exist within an S3 bucket

#### Distribution Configurations

- You must specify your origin location
- Select specific caching behavioral options
- Define the distribution settings (which edge locations you want your data to be distributed to)

#### CloudFront & WAF

- Web Application Firewall (WAF)
- WAF provides additional security for your web application tier
- Encryption can be applied through SSL certificates

#### High Level Process

- When content from your website is accessed, the end user will be directed to their closest edge location in terms of
latency, to see if the content is cached by CloudFront at that edge location.
- If the content is there, the user will access the content from the edge location instead of the origin, therefore
reducing latency.
- If the content is not there, or the cache has expired for that content at the edge location, then CloudFront will
request the content from the source origin again. This content will then be used to maintain a fresh cache for any
future request until it again expires.

#### Pricing

[CloudFront Pricing](https://aws.amazon.com/cloudfront/pricing/)

<br/>

## AWS Data Services

<br/>

### AWS Storage Gateway

**Storage Gateway** allows you to provide a gateway between your own data center's storage systems, such as your
SAN, NAS, or DAS, and Amazon S3 in Glacier on AWS.

The storage gateway itself is a software appliance that can be installed within your own data center, which allows
integration between your on-premise storage and AWS. This connectivity can allow you to scale your storage requirements
both securely and cost efficiently.

The software appliance can be downloaded from AWS as a virtual machine.

*3 configurations* available:
1. File Gateway
2. Volume Gateways
3. Tape Gateway

#### File Gateway

- Allows you to securely store your files as objects in S3
- Ability to mount or map drives to an S3 bucket as if it was a share held locally

When storing files using the file gateway, they are sent to S3 over HTTPS, and are also encrypted with S3's own
server-side encryption, SSE-S3. In addition to this, a local on-premise cache is also provisioned for accessing your
most recently accessed files to optimize latency, which also helps to reduce egress traffic costs. When your file
gateway is first configured, you must associate it with your S3 Bucket which the gateway will then present as an
NFS v3 or v41 file system to your internal applications. This allows you to view the Bucket as a normal NFS file system,
making it easier to mount as a drive in Linux or map a drive to it in Microsoft. Any files that are then written to
these NFS file systems are stored in S3 as individual objects as a one to one mapping of files to objects.

#### Volume Gateways

Can be configured in 2 ways:
1. Stored Volume Gateways
2. Cached Volume Gateways

#### Stored Volume Gateways

- Used to backup your local storage volumes to Amazon S3
- Your entire local data set remains on-premise ensuring low latency data access

Volumes created and configured within the storage gateway are backed by Amazon S3, and are mounted as iSCSI devices
that your applications can then communicate with. During the volume creation, these are mapped to your on premise
storage devices, which can either hold existent data or be a new disk. As data is written to these iSCSI devices, the
data is actually written to your local storage services such as your own NAS, SAN, or DAS storage solution. However,
the storage gateway then asynchronously copies this data to Amazon S3 as EBS snapshots. Having your entire data set
remain locally ensures you have the lowest latency possible to access your data, which may be required for specific
applications, or security compliance and governance controls whilst at the same time, providing a backup solution
which is governed by the same controls and security that S3 offers.

- Volumes can be between one gig and 16 terabytes
- Each storage gateway, up to 32 stored volumes
- Maximum total of 512 terabytes of storage per gateway
- A storage buffer using on-premise storage is used as a staging point for data that is waiting to be written to S3
- Data is uploaded across an SSL channel and stored in an encrypted format in S3
- Snapshots can be taken of volumes at any point an stored as EBS snapshots in S3
  - These snapshots are incremental ensuring that only the data that's changed since the last backup is copied,
  helping to minimize storage costs in S3

Gateway stored volumes makes **disaster recovery** very simple. For example, consider the
scenario that you lost your local application and storage layers on premise. Providing you had prevision for such a
situation, you may have AMI templates that mimic your application tier which you could prevision as EC2 instances
within AWS. You could then attach EBS volumes to these instances which could be created from your storage gateway
volume snapshots, which would be stored on S3, giving you access to your production data required. Your application
storage infrastructure could be potentially up and running again in a matter of minutes within a VPC with connectivity
from your on-premise data center.

#### Cached Volume Gateways

Cached volume gateways are different to stored volume gateways, in that:
- The primary data storage is actually Amazon S3 (rather than your own local storage solution).
- Local data storage is used for buffering and a local cache for recently accessed data.

Again, during the creation of these volumes, they are presented as iSCSI volumes which can be mounted by your application
servers. The volumes themselves are backed by the Amazon S3 infrastructure as opposed to your local disks as seen in
the stored volume gateway deployment. As a part of this volume creation, you must also select some local disks
on-premise to act as your local cache and a buffer for data waiting to be uploaded to S3. Again, this buffer is used
as a staging point for data that is waiting to be written to S3 and during the upload process the data is encrypted
using an SSL channel, where the data is then encrypted within SSE-S3.

The limitations are slightly different with cached volume gateways, in that:
- each volume created can be up to 32 terabytes in size.
- support for up to 32 volumes
- total storage capacity of 1024 terabytes per cache volume gateway.
- still possible to take incremental backups with these volumes as EBS snapshots.


#### Tape Gateway

AKA Gateway VTL, Virtual Tape Library.

This allows you to, again,
- backup your data to S3 from your own corporate data center,
- but also leverage *Amazon Glacier* for data archiving.

VTL is essentially a cloud based tape backup solution, replacing physical components with virtual ones. This
functionality allows you to use your existing tape backup application infrastructure within AWS, providing a more
robust and secure backup and archiving solution.

**VTL Elements**:

- *Storage gateway*
    - The gateway itself is configured as a tape gateway, which has a capacity to hold 1500 virtual tapes.
- *Virtual tapes*
    - These are a virtual equivalent to a physical backup tape cartridge which can be anything from 100 gig to 2.5
    terabytes in size, and any data stored on the virtual tapes are backed by AWS S3 and appear in the virtual tape
    library.
- *Virtual tape library, VTL*
    - As you may have guessed, these are a virtual equivalent to a tape library that contain virtual tapes.
- *Tape drives*
  - Every VTL comes with 10 virtual tape drives, which are presented to your backup applications as iSCSI devices.
- *Media changer*
    - This is a virtual device that manages tapes to and from the tape drive to your VTL, and again is presented as an
    iSCSI device to your backup applications.
- *Archive*
    - Equivalent to an off-site tape backup storage facility where you can archive tapes from your virtual
    tape library to Amazon Glacier. If retrieval of the tapes are required, storage gateway uses the standard retrieval
    option which can take between three to five hours to retrieve your data.

Once your storage gateway has been configured as a tape gateway:
- your applications and backup software can mount the tape drives along with the media changer as iSCSI devices to
make the connection.
- You can then create the required virtual tapes as you need them for backup and your backup software can use these
to backup the required data which is stored on S3.
- Archiving tapes moves data from S3 to Glacier.

<br/>

### AWS Snowball

#### AWS Snowball

- A physical device shipped from AWS to help with large scale data transfers.
- The devices are literally loaded up with your data (or you load it up if getting data to S3) and shipped to you so
that you can then take the data from them to your data center.
- Used to securely transfer large amounts of data in and out of AWS (Petabyte Scale)
- Either from your on-premise data center to S3, or form S3 to your data center.
- The snowball appliance comes as either a 50 TB or 80 TB device.
- The snowball appliance is dust, water and tamper resistant.
- Built for high speed data transfer:
    - RJ45 (Cat6)
    - SFP+ Copper
    - SFP+ Optical

#### Encryption and Tracking

- By default, all data transferred to a snowball appliance is automatically encrypted using 256-bit encryption
keys generated from KMS, the Key Management Service.
- End to end tracking using an E-ink shipping label.
- Assists with the delivery to the correct AWS premises.
- The snowball appliance can be tracked with SNS (Simple Notification Service) text messages or via the
AWS Management Console.
- AWS Snowball is also HIPAA compliant allowing the transfer of protected health data into and out of S3.
- Data removal from the appliance is the responsibility of AWS, conforming to NIST standards.

#### Data Aggregation

When sending or retrieving data, snowball appliances can be aggregated together. For example, if you needed to
retrieve 400 terabytes of data from S3 then your data will be sent by five 80 terabyte snowball appliances.

So, from a disaster recovery perspective when might you need to use AWS Snowball?
- Well it all depends on how much data you need to get back from S3 to your own corporate data center and how
quickly you can do that.
- On the other hand, how much data you need to get into S3.
- This'll depend on the connection you have to AWS from your data center. You may have direct-connect connections, a
VPN, or just an internet connection. And if you need to restore multiple petabytes of data, this could take weeks
or even months to complete.

*As a general rule, if your data retrieval will take longer than a week using your existing connection method, then
you should consider using AWS Snowball.*

#### AWS Snowball Process

1. Create an Expert Job
2. Receive delivery of your appliance
3. Connect the appliance to your network
    - Connect the appliance to your network when the appliance is off
    - Power on the appliance
    - Configure the network settings
4. You are now ready to transfer the data!
    - Access the required credentials
    - Install the Snowball Client
    - Transfer the data using the client
    - Disconnect the appliance when the data transfer is complete.
5. Return the snowball appliance to AWS.

#### Pricing

[AWS Snowball Pricing](https://aws.amazon.com/snowball/pricing/)

<br/>

## Course Summary

**Amazon S3**

- Fully managed object based storage service: highly available, highly durable, cost effective and widely accessible.
- Almost unlimited storage capabilities
- The smallest file size supported = 0 bytes and the largest = 5 terabytes
- Data is uploaded to a specific region and duplicated across multiple AZs automatically
- Objects have a durability of 99.999999999% and an availability of 99.99%
- Objects are stored in buckets or folders within a bucket
- S3 has 3 storage classes
    - Standard
    - Standard - Infrequent Access
    - Reduced Redundancy
- Security features
    - Bucket policies
    - Access control lists
    - Data encryption
    - SSL support
- Data management features include versioning and lifecycle rules
- Often used for data backup, static websites and large datasets
- Integrates with other AWS services
- Pricing primarily based on the amount of storage used, plus request and data transfer costs

**Amazon Glacier**

- An extremely low cost, long term, durable storage solution ideally suited for long term backup and archival
requirements
- 99.999999999% durability
- Much cheaper than S3
- Does NOT provide instant access of data retrieval
- Data structure is centered around *Vaults* and *Archives*
- A *Vault* simply acts as a container for *Archives*
- An *Archive* is essentially your data stored within *Vaults*
- Unlimited *Archives* available
- The Glacier console only allows you to create *Vaults*
- You must use the Glacier web service API or the AWS SDKs to move data in/out of Glacier
- There are 3 different retrieval options: Expedited, Standard and Bulk
- Encryption is enabled by default using the AES-256 encryption algorithm
- Access control is governed through IAM, Vault access policies and Vault lock policies
- Pricing remains the same regardless of how much storage is used. however there is still request, data transfer and
data retrieval costs
- Designed to archive data for extended periods of time (Cold Storage) for a very small cost

**EC2 Instance Storage**

- Volumes physically reside on the same host that provides your EC2 instance
- Amazon EC2 Instance store volumes act as local drives to an EC2 Instance
- Instance store volumes provide ephemeral (temporary) storage for you EC2 instances.
  - Ephemeral storage means that the block level storage that it provides offers no means of persistency.
    Any data stored on these volumes is considered temporary
- Not recommended for critical or valuable data
- If your instance is either stopped or terminated
  - Any data note stored on that instance store volume associated with this instance will be deleted without any
    means of recovery.
- If your instance was simply rebooted, your data would remain intact.
- Instance store volumes are not available for all instances
- Capacity of instance store volumes increases with the size of the EC2 instance
- Instance store volumes have the same security mechanisms provided by EC2
  - They are not a separate service from EC2
- No additional cost for storage; it's included in the price of the EC2 instance.
- Offer a very high I/O speed
- Instance store volumes are ideal as a cache or buffer for rapidly changing data without need for retention
- Often used within a load balancing group, where data is replicated and pooled across the fleet
- Instance store volumes should not be used for:
  - Data that needs to remain persistent
  - Data that needs to be accessed and shared by others
- If you need to use block level storage and want to maintain persistency, EBS is recommended.

**Elastic Block Store (EBS)**

- Provides block level storage to your EC2 instances
- Offers persistent and durable data storage
- EBS volumes can be attached and detached from instances
- Primarily used for data that is rapidly changing
- A single EBS volume can only be attached to a single EC2 instance
- Multiple EBS volumes can only be attached to a single EC2 instance
- EBS snapshots provide a point in time backup of the volume and are stored in S3
- You can create a new volume from a snapshot
- All writes are replicated multiple times within a single AZ
- EBS volumes are only available in a single AZ
- 4 types of EBS Volumes available:
    - 2 that are SSD backed:
        - General Purpose SSD (GP2)
        - Provisioned IOPS SSD (IO1)
    - 2 that are HDD backed:
        - Cold HDD (SC1)
        - Throughput Optimized HDD (ST1)
    - Cost depends on type
- Storage provisioned is billed to you on a per-second basis
- EBS snapshots also incur S3 storage costs
- EBS encrypts data both at rest and when in transit if required
- Encrypted volumes will also encrypt snapshots automatically

**Elastic File System (EFS)**

- Provides file level storage
- Fully managed, highly available and durable
- Allows you to create shared file systems
- Can meet demands by thousands of concurrent EC2 instances
- Limitless capacity
- Storage capacity grows with use
- Ideal for applications that scale across multiple instances
- EFS is a regional service
- Designed to maintain a high level of throughput (MB/s) and low latency access response
- Mount targets allow connectivity for you instances to your EFS
- Only compatible with NFS V4.0 and V4.1
- Does not support the Windows OS
- Linux instances must have the NFS Client installed for the mounting process
- EFS can run in 2 different performance mode of operations:
    - General Purpose (default)
        - Used for most cases
        - Provides the lowest latency
        - Maximum of 7000 file system operations per second for your EFS
    - Max I/O
        - Used for huge scale architectures
        - Concurrent access of 1000's of instances
        - Can exceed 7000 file system operations per second
        - Virtually unlimited amount of throughput and IOPS
        - There is however an additional latency to each I/O
- Encryption at rest is possible with KMS
- Encryption is transit is not supported
- File Sync can be used to migrate data to EFS via an agent
- Pricing is charged at per GB-months

**Amazon CloudFront**

- Amazon CloudFront is a content delivery network (CDN)
- Distributes your source web data closer to the end user via AWS edge locations as cached data
- It doesn't provide durability of data
- AWS edge locations are sites deployed across the globe to chance data and reduce latency
- Distributions control which source data it needs to distribute and to where
- Methods of data distribution:
    - Web Distribution
    - RTMP Distribution
- Distributions require origins that contain your source data, such as S3
- Data can be distributed using the following edge location options:
    - US, Canada and Europe
    - US, Canada, Europe and Asia
    - All edge locations (best performance)
- Amazon CloudFront can interact with the Web Applications Firewall service for additional security and
web application protection
- SSL certificates can be configured to be used with the distribution
- Pricing is primarily based on data transfer assets and HTTP requests

**AWS Storage Gateway**

- Allows you to provide a gateway between your own data center storage and Amazon S3/Glacier on AWS
- The storage gateway is a software appliance
- Storage Gateway offers File, Volume and Tape Gateway configurations
  - File Gateway
    - Securely store your files as objects within S3
    - Mount or map drives to an S3 bucket
    - A local on premise cache is provisioned most recently accessed files
  - Volume Gateways
    - Stored Volume Gateways
      - Used to backup local storage volumes to S3
      - Your data library also remains locally on premise
      - Presented as iSCSI volumes
    - Cached-Volume Gateways
      - Primary data storage is Amazon S3 rather than your own local storage solution
      - Utilize local data storage as a buffer and a cache
      - Presented as iSCSI volumes
- Pricing is based upon storage usage, requests and data transfer

**AWS Snowball**

- Used to securely transfer large amounts of data in and out of AWS using physical appliance, known as
a snowball
- The snowball comes as either a 50TB or 80TB storage device
- It is dust, water and tamper resistant
- Designed to allow for high speed data transfer
- All data transferred to the Snowball appliance is automatically encrypted by default
- HIPPA Compliant
- AWS most ensure the data held on the snowball appliance is deleted and removed
- Snowballs can be aggregated together
- If your data retrieval will take longer than a week consider using AWS Snowball
- Pricing is based on Amazon S3 data charges plus additional costs for the data transfer job shipping

---
